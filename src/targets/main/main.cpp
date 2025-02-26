#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <SuiteSparseQR.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "config/utils.hpp"
#include "map/map.hpp"
#include "map/utils.hpp"
#include "optimization/objective.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "scan/scene.hpp"
#include "scan/shape/shape.hpp"
#include "scan/utils.hpp"
#include "state/state.hpp"
#include "visualize/utils.hpp"

// Converts Eigen::SparseMatrix to CHOLMOD sparse matrix
void eigen_to_cholmod(const Eigen::SparseMatrix<double> &eigenMatrix,
                      cholmod_sparse *&cholmodMatrix,  //
                      cholmod_common &c) {
  cholmodMatrix = cholmod_allocate_sparse(eigenMatrix.rows(), eigenMatrix.cols(),
                                          eigenMatrix.nonZeros(), true, true, -1, CHOLMOD_REAL, &c);
  if (!cholmodMatrix) {
    throw std::runtime_error("Failed to allocate CHOLMOD sparse matrix.");
  }

  auto *x = static_cast<double *>(cholmodMatrix->x);
  auto *i = static_cast<int *>(cholmodMatrix->i);
  auto *p = static_cast<int *>(cholmodMatrix->p);

  int k = 0;
  for (int col = 0; col < eigenMatrix.outerSize(); ++col) {
    p[col] = k;
    for (Eigen::SparseMatrix<double>::InnerIterator it(eigenMatrix, col); it; ++it) {
      i[k] = it.row();
      x[k++] = it.value();
    }
  }
  p[eigenMatrix.cols()] = k;
}

// Runs the optimization process
void run_optimization(ObjectiveFunctor<2> &functor, Eigen::VectorXd &params,  //
                      cholmod_common &c,                                      //
                      const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                      const MapArgs<2> &map_args,
                      const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
                      const OptimizationArgs &opt_args,  //
                      const VisualizationArgs &vis_args) {
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
    cholmod_dense *rhs =
        cholmod_allocate_dense(jt_residuals.size(), 1, jt_residuals.size(), CHOLMOD_REAL, &c);
    std::memcpy(rhs->x, jt_residuals.data(), jt_residuals.size() * sizeof(double));

    // LHS of the linear system
    Eigen::SparseMatrix<double> jt_j = jacobian_sparse.transpose() * jacobian_sparse;
    Eigen::SparseMatrix<double> identity(jt_j.rows(), jt_j.cols());
    identity.setIdentity();
    Eigen::SparseMatrix<double> lhs_eigen = jt_j + lambda * identity;

    // Cholesky factorization
    cholmod_sparse *lhs;
    eigen_to_cholmod(lhs_eigen, lhs, c);
    cholmod_factor *lhs_factorised = cholmod_analyze(lhs, &c);
    cholmod_factorize(lhs, lhs_factorised, &c);

    // Solve the linear system
    cholmod_dense *cholmod_result = cholmod_solve(CHOLMOD_A, lhs_factorised, rhs, &c);
    if (!cholmod_result) {
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&lhs, &c);
      throw std::runtime_error("cholmod_solve failed.");
    }

    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x, jt_residuals.size() * sizeof(double));
    params += delta;

    // Update lambda based on error improvement
    functor(params, residuals);
    double new_error = residuals.squaredNorm();
    lambda = new_error < error ? lambda / opt_args.lambda_factor : lambda * opt_args.lambda_factor;
    error = new_error;

    // Logging
    const double update_norm = delta.norm();
    if (opt_args.std_out) {
      std::cout << "Iteration: " << iter << " Error: " << error << " Update Norm: " << update_norm
                << std::endl;
    }
    if (opt_args.visualize) {
      visualize_map(params, scans, map_args, initial_frame, vis_args);
    }

    // Free memory
    cholmod_free_sparse(&lhs, &c);
    cholmod_free_dense(&rhs, &c);
    cholmod_free_dense(&cholmod_result, &c);

    // Early stopping
    if (update_norm < opt_args.tolerance) break;
  }
}

int main() {
  try {
    // Setup and read configuration
    Args<2> args = setup_from_yaml<2>("../config/sdf.yml");
    Scans scans = read_scans(args.general_args.data_path);
    const std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = scans.scans;

    // Initialize the map and set up the optimization parameters
    Map<2> map(args.map_args);
    if (args.general_args.from_ground_truth) {
      const std::filesystem::path scene_path = args.general_args.data_path / "scene_info.txt";
      const Scene scene = Scene::from_file(scene_path);
      map.from_ground_truth(scene);
    } else {
      map.set_value(args.general_args.initial_value);
    }
    Eigen::VectorXd params = flatten<2>(State<2>(map, scans.frames));

    // Visualize the initial map
    visualize_map(params, point_clouds, args.map_args, scans.frames[0], args.vis_args);

    // Set up the objective functor for optimization
    int num_points = 0;
    for (const auto &cloud : point_clouds) {
      num_points += cloud.size();
    }

    const uint num_smoothing_residuals =
        args.objective_args.smoothness_derivative_type == DerivativeType::FORWARD
            ? (map.get_num_points(0) - 1) * (map.get_num_points(1) - 1)
            : map.total_points();
    const int num_residuals =
        num_points * (args.objective_args.scanline_points + 1) + num_smoothing_residuals;
    ObjectiveFunctor<2> functor(params.size(), num_residuals, args.map_args, point_clouds,
                                args.objective_args, scans.frames[0]);

    // Run the optimization process
    cholmod_common c;
    cholmod_start(&c);
    run_optimization(functor, params, c, point_clouds, args.map_args, scans.frames[0],
                     args.optimization_args, args.vis_args);
    cholmod_finish(&c);

    // Visualize the optimized map
    visualize_map(params, point_clouds, args.map_args, scans.frames[0], args.vis_args);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
