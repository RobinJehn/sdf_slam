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
void runOptimization(ObjectiveFunctor<2> &functor, Eigen::VectorXd &params,
                     cholmod_common &c,
                     const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                     int map_points_x, int map_points_y, bool visualize) {
  Eigen::SparseMatrix<double> jacobian_sparse;
  Eigen::VectorXd residuals(functor.values());
  double lambda = 1;
  int max_iters = 20;
  double tolerance = 1e-5;

  for (int iter = 0; iter < max_iters; ++iter) {
    functor.sparse_df(params, jacobian_sparse);
    functor(params, residuals);

    std::cout << "Iteration: " << iter << " Error: " << residuals.squaredNorm()
              << std::endl;

    if (jacobian_sparse.nonZeros() == 0) {
      std::cerr << "Error: Jacobian has no non-zero entries." << std::endl;
      return;
    }

    Eigen::SparseMatrix<double> jt_j =
        jacobian_sparse.transpose() * jacobian_sparse;

    Eigen::SparseMatrix<double> identity(jt_j.rows(), jt_j.cols());
    identity.setIdentity();
    Eigen::SparseMatrix<double> lm_matrix = jt_j + lambda * identity;
    Eigen::VectorXd jt_residuals = -jacobian_sparse.transpose() * residuals;

    cholmod_sparse *cholmod_lm_matrix;
    eigenToCholmod(lm_matrix, cholmod_lm_matrix, c);

    cholmod_dense *rhs = cholmod_allocate_dense(
        jt_residuals.size(), 1, jt_residuals.size(), CHOLMOD_REAL, &c);
    std::memcpy(rhs->x, jt_residuals.data(),
                jt_residuals.size() * sizeof(double));

    cholmod_factor *L = cholmod_analyze(cholmod_lm_matrix, &c);
    cholmod_factorize(cholmod_lm_matrix, L, &c);
    cholmod_dense *cholmod_result = cholmod_solve(CHOLMOD_A, L, rhs, &c);

    if (!cholmod_result) {
      std::cerr << "Error: cholmod_solve failed." << std::endl;
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&cholmod_lm_matrix, &c);
      return;
    }

    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x,
                jt_residuals.size() * sizeof(double));
    params += delta;

    double prev_error = residuals.squaredNorm();
    functor(params, residuals);
    double new_error = residuals.squaredNorm();

    if (delta.norm() < tolerance)
      break;
    lambda = new_error < prev_error ? lambda * 0.3 : lambda * 3;

    cholmod_free_dense(&rhs, &c);
    cholmod_free_dense(&cholmod_result, &c);
    cholmod_free_sparse(&cholmod_lm_matrix, &c);

    if (visualize) {
      visualizeMap(params, scans, {map_points_x, map_points_y}, {-3, -8},
                   {2 * M_PI + 3, 4});
    }
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

  // Initialize the map and set up the optimization parameters
  Map<2> map =
      init_map(-3, 2 * M_PI + 3, -8, 4, map_points_x, map_points_y, false);
  ObjectiveFunctor<2> functor(6 + map_points_x * map_points_y,
                              (scans.first->size() + scans.second->size()) * 21,
                              {map_points_x, map_points_y}, {-3, -8},
                              {2 * M_PI + 3, 4}, point_clouds, 20, true, 0.1);

  Eigen::Transform<double, 2, Eigen::Affine> initial_frame =
      Eigen::Translation<double, 2>(pos1) * Eigen::Rotation2D<double>(theta1);
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_2 =
      Eigen::Translation<double, 2>(pos2) * Eigen::Rotation2D<double>(theta2);
  Eigen::VectorXd params =
      flatten<2>(State<2>(map, {initial_frame, initial_frame_2}));

  cholmod_common c;
  cholmod_start(&c);
  runOptimization(functor, params, c, point_clouds, map_points_x, map_points_y,
                  false);
  cholmod_finish(&c);

  // Visualize the optimized map and transformed point clouds
  visualizeMap(params, point_clouds, {map_points_x, map_points_y}, {-3, -8},
               {2 * M_PI + 3, 4});

  return 0;
}
