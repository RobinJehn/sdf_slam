#include "config/utils.hpp"
#include "map/map.hpp"
#include "map/utils.hpp"
#include "optimization/objective.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "scan/generate.hpp"
#include "scan/utils.hpp"
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
    throw std::runtime_error("Failed to allocate CHOLMOD sparse matrix.");
  }

  auto *x = static_cast<double *>(cholmodMatrix->x);
  auto *i = static_cast<int *>(cholmodMatrix->i);
  auto *p = static_cast<int *>(cholmodMatrix->p);

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

// Runs the optimization process
void runOptimization(
    ObjectiveFunctor<2> &functor, Eigen::VectorXd &params, cholmod_common &c,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
    const MapArgs<2> &map_args, const bool visualize,
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
      throw std::runtime_error("Jacobian has no non-zero entries.");
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
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&lhs, &c);
      throw std::runtime_error("cholmod_solve failed.");
    }

    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x,
                jt_residuals.size() * sizeof(double));
    params += delta;

    // Update lambda based on error improvement
    functor(params, residuals);
    double new_error = residuals.squaredNorm();
    lambda = new_error < error ? lambda / opt_args.lambda_factor
                               : lambda * opt_args.lambda_factor;
    error = new_error;

    // Logging
    const double update_norm = delta.norm();
    std::cout << "Iteration: " << iter << " Error: " << error
              << " Update Norm: " << delta.norm() << std::endl;
    if (visualize) {
      visualizeMap(params, scans, map_args, initial_frame);
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
  try {
    Scans scans = read_scans("../data/scans");
    const std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = scans.scans;

    const bool visualize = true;
    const bool from_ground_truth = true;

    Args<2> args = setup_from_yaml<2>("../config/sdf.yml");

    const int num_points = point_clouds[0].size() * point_clouds.size();
    const int num_residuals =
        num_points * (args.objective_args.scanline_points + 1);

    // Initialize the map and set up the optimization parameters
    Map<2> map = init_map(args.map_args, from_ground_truth);

    Eigen::VectorXd params = flatten<2>(State<2>(map, scans.frames));
    ObjectiveFunctor<2> functor(params.size(), num_residuals, args.map_args,
                                point_clouds, args.objective_args,
                                scans.frames[0]);

    cholmod_common c;
    cholmod_start(&c);
    runOptimization(functor, params, c, point_clouds, args.map_args, visualize,
                    scans.frames[0], args.optimization_args);
    cholmod_finish(&c);

    // Visualize the optimized map and transformed point clouds
    visualizeMap(params, point_clouds, args.map_args, scans.frames[0]);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
