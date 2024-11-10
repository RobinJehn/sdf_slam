#include "map/map.hpp"
#include "optimization/objective.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "scan/generate.hpp"
#include "state/state.hpp"
#include "visualize/utils.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <SuiteSparseQR.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>

// Converts Eigen::SparseMatrix to CHOLMOD sparse matrix
void eigenToCholmod(const Eigen::SparseMatrix<double> &eigenMatrix,
                    cholmod_sparse *&cholmodMatrix, cholmod_common &c) {
  cholmodMatrix = cholmod_allocate_sparse(
      eigenMatrix.rows(), eigenMatrix.cols(), eigenMatrix.nonZeros(), true,
      true, -1, CHOLMOD_REAL, &c);
  if (!cholmodMatrix) {
    std::cerr << "Error: Failed to allocate cholmod sparse matrix."
              << std::endl;
    return;
  }

  double *x = static_cast<double *>(cholmodMatrix->x);
  int *i = static_cast<int *>(cholmodMatrix->i);
  int *p = static_cast<int *>(cholmodMatrix->p);

  int k = 0;
  for (int col = 0; col < eigenMatrix.outerSize(); ++col) {
    p[col] = k;
    for (Eigen::SparseMatrix<double>::InnerIterator it(eigenMatrix, col); it;
         ++it) {
      i[k] = it.row();
      x[k++] = it.value();
    }
  }
  p[eigenMatrix.cols()] = k;
}

// Setup initial scan positions and angles
void setupScanPositions(double &theta1, double &theta2, Eigen::Vector2d &pos1,
                        Eigen::Vector2d &pos2) {
  theta1 = 9 * M_PI / 16;
  pos1 = {3.5, -5};
  theta2 = 8 * M_PI / 16;
  pos2 = {4, -5};
}

// Runs the optimization process
void runOptimization(
    ObjectiveFunctor<2> &functor, Eigen::VectorXd &params, cholmod_common &c,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
    const int map_points_x, const int map_points_y, const bool visualize,
    const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
    const OptimizationArgs &opt_args) {
  Eigen::SparseMatrix<double> jacobian_sparse;
  Eigen::VectorXd residuals(functor.values());
  double lambda = opt_args.initial_lambda;

  functor(params, residuals);
  double error = residuals.squaredNorm();
  std::cout << "Initial Error: " << error << std::endl;

  for (int iter = 0; iter < opt_args.max_iters; ++iter) {
    // Compute the Jacobian matrix
    functor.sparse_df(params, jacobian_sparse);
    if (jacobian_sparse.nonZeros() == 0) {
      std::cerr << "Error: Jacobian has no non-zero entries." << std::endl;
      return;
    }

    // RHS of the linear system
    Eigen::VectorXd jt_residuals = -jacobian_sparse.transpose() * residuals;
    cholmod_dense *rhs = cholmod_allocate_dense(
        jt_residuals.size(), 1, jt_residuals.size(), CHOLMOD_REAL, &c);
    std::memcpy(rhs->x, jt_residuals.data(),
                jt_residuals.size() * sizeof(double));

    // LHS of the linear system
    Eigen::SparseMatrix<double> jt_j =
        jacobian_sparse.transpose() * jacobian_sparse;
    Eigen::SparseMatrix<double> identity(jt_j.rows(), jt_j.cols());
    identity.setIdentity();
    Eigen::SparseMatrix<double> lhs_eigen = jt_j + lambda * identity;

    // Cholesky factorization
    cholmod_sparse *lhs;
    eigenToCholmod(lhs_eigen, lhs, c);
    cholmod_factor *lhs_factorised = cholmod_analyze(lhs, &c);
    cholmod_factorize(lhs, lhs_factorised, &c);

    // Solve the linear system
    cholmod_dense *cholmod_result =
        cholmod_solve(CHOLMOD_A, lhs_factorised, rhs, &c);

    if (!cholmod_result) {
      std::cerr << "Error: cholmod_solve failed." << std::endl;
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&lhs, &c);
      return;
    }

    // Update the parameters
    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x,
                jt_residuals.size() * sizeof(double));
    params += delta;

    // Update lambda
    functor(params, residuals);
    double new_error = residuals.squaredNorm();
    lambda = new_error < error ? lambda / opt_args.lambda_factor
                               : lambda * opt_args.lambda_factor;
    error = new_error;

    // Logging
    const double update_norm = delta.norm();
    std::cout << "Iteration: " << iter << " Error: " << error
              << " Update Norm:" << update_norm << std::endl;
    if (visualize) {
      visualizeMap(params, scans, {map_points_x, map_points_y}, {-3, -8},
                   {2 * M_PI + 3, 4}, initial_frame);
    }

    // Free memory
    cholmod_free_sparse(&lhs, &c);
    cholmod_free_dense(&rhs, &c);
    cholmod_free_dense(&cholmod_result, &c);

    // Early stopping
    if (update_norm < opt_args.tolerance)
      break;
  }
}

int main() {
  double theta1, theta2;
  Eigen::Vector2d pos1, pos2;
  setupScanPositions(theta1, theta2, pos1, pos2);

  auto scans = create_scans(pos1, theta1, pos2, theta2);
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = {*scans.first,
                                                             *scans.second};

  const int map_points_x = 200;
  const int map_points_y = 200;

  const double x_min = -3;
  const double x_max = 2 * M_PI + 3;
  const double y_min = -8;
  const double y_max = 4;
  const int number_of_points = 20;
  const bool both_directions = true;
  const double step_size = 0.1;
  const bool visualize = false;
  const bool from_ground_truth = true;

  OptimizationArgs optimization_args;
  optimization_args.max_iters = 100;
  optimization_args.initial_lambda = 1;
  optimization_args.tolerance = 1e-5;
  optimization_args.lambda_factor = 1;

  // Initialize the map and set up the optimization parameters
  Map<2> map = init_map(x_min, x_max, y_min, y_max, map_points_x, map_points_y,
                        from_ground_truth);

  Eigen::Transform<double, 2, Eigen::Affine> initial_frame =
      Eigen::Translation<double, 2>(pos1) * Eigen::Rotation2D<double>(theta1);
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_2 =
      Eigen::Translation<double, 2>(pos2) * Eigen::Rotation2D<double>(theta2);

  const std::vector<Eigen::Transform<double, 2, Eigen::Affine>>
      transformations = {initial_frame, initial_frame_2};

  ObjectiveFunctor<2> functor(
      (transformations.size() - 1) * 3 + map_points_x * map_points_y,
      (scans.first->size() + scans.second->size()) * (number_of_points + 1),
      {map_points_x, map_points_y}, {x_min, y_min}, {x_max, y_max},
      point_clouds, number_of_points, both_directions, step_size,
      initial_frame);

  Eigen::VectorXd params = flatten<2>(State<2>(map, transformations));

  cholmod_common c;
  cholmod_start(&c);
  runOptimization(functor, params, c, point_clouds, map_points_x, map_points_y,
                  visualize, initial_frame, optimization_args);
  cholmod_finish(&c);

  // Visualize the optimized map and transformed point clouds
  visualizeMap(params, point_clouds, {map_points_x, map_points_y},
               {x_min, y_min}, {x_max, y_max}, initial_frame);

  return 0;
}
