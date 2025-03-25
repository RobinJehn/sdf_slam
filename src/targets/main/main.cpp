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
                      const Args<2> &args,  //
                      const std::string &exp_name) {
  Eigen::SparseMatrix<double> jacobian_sparse;
  Eigen::VectorXd residuals(functor.values());

  OptimizationArgs opt_args = args.optimization_args;
  VisualizationArgs vis_args = args.vis_args;
  ObjectiveArgs obj_args = args.objective_args;
  GeneralArgs gen_args = args.general_args;
  double lambda = opt_args.initial_lambda;

  functor(params, residuals);
  double error = residuals.squaredNorm();
  std::cout << "Initial Error: " << error << std::endl;

  if (gen_args.save_results) {
    // Save the initial parameters to a file
    std::ofstream params_file(exp_name + "/params_0.txt");
    if (params_file.is_open()) {
      params_file << params << std::endl;
      params_file.close();
    } else {
      throw std::runtime_error("Unable to open file to save initial parameters.");
    }
  }

  for (int iter = 0; iter < opt_args.max_iters; ++iter) {
    // Compute the Jacobian matrix
    auto start_time_jac = std::chrono::high_resolution_clock::now();
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
    auto end_time_jac = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span_jac = end_time_jac - start_time_jac;

    if (gen_args.save_results) {
      // Save the parameters to a file
      std::ofstream params_file(exp_name + "/params_" + std::to_string(iter + 1) + ".txt");
      if (params_file.is_open()) {
        params_file << params << std::endl;
        params_file.close();
      } else {
        throw std::runtime_error("Unable to open file to save initial parameters.");
      }
    }

    // Update lambda based on error improvement
    auto start_time_res = std::chrono::high_resolution_clock::now();
    functor(params, residuals);
    double new_error = residuals.squaredNorm();
    auto end_time_res = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span_res = end_time_res - start_time_res;

    lambda = new_error < error ? lambda / opt_args.lambda_factor : lambda * opt_args.lambda_factor;
    error = new_error;

    // Logging
    const double update_norm = delta.norm();
    if (vis_args.std_out) {
      std::cout << "Iteration: " << iter << ", Error: " << error << ", Update Norm: " << update_norm
                << ", Jacobian computation time: " << time_span_jac.count()
                << " ms, Residual computation time: " << time_span_res.count() << " ms"
                << std::endl;
    }
    if (vis_args.visualize || vis_args.save_file) {
      visualize_map(params, scans, map_args, initial_frame, vis_args, obj_args, vis_args.visualize,
                    vis_args.save_file, exp_name);
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
    visualize_map(params, point_clouds, args.map_args, scans.frames[0], args.vis_args,
                  args.objective_args, /* visualize*/ args.vis_args.initial_visualization,
                  /* save_file */ false);

    // Set up the objective functor for optimization
    int num_points = 0;
    for (const auto &cloud : point_clouds) {
      num_points += cloud.size();
    }

    uint num_smoothing_residuals;
    switch (args.objective_args.smoothness_derivative_type) {
      case DerivativeType::FORWARD:
        num_smoothing_residuals = (map.get_num_points(0) - 1) * (map.get_num_points(1) - 1);
        break;
      case DerivativeType::CENTRAL:
        num_smoothing_residuals = (map.get_num_points(0) - 2) * (map.get_num_points(1) - 2);
        break;
      case DerivativeType::UPWIND:
        num_smoothing_residuals = map.total_points();
        break;
      default:
        num_smoothing_residuals = map.total_points();
    }

    std::vector<Eigen::Transform<double, 2, Eigen::Affine>> odometry;
    for (int i = 0; i < scans.frames.size() - 1; ++i) {
      odometry.push_back(scans.frames[i].inverse() * scans.frames[i + 1]);
    }
    const uint num_odometry_residuals = odometry.size() * 3;  // 3 per odometry reading

    const int num_residuals = num_points * (args.objective_args.scanline_points + 1) +
                              num_smoothing_residuals + num_odometry_residuals;

    ObjectiveFunctor<2> functor(params.size(), num_residuals, args.map_args, point_clouds, odometry,
                                args.objective_args, scans.frames[0]);

    // Save the parameters to a YAML file
    std::string exp_name =
        "../experiments/experiment_" +
        std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::filesystem::create_directories(exp_name);
    const ObjectiveArgs &obj_args = args.objective_args;
    std::ofstream yaml_file(exp_name + "/params.yml");
    yaml_file << "scan_line_factor: " << obj_args.scan_line_factor << "\n";
    yaml_file << "scanline_points: " << obj_args.scanline_points << "\n";
    yaml_file << "step_size: " << obj_args.step_size << "\n";
    yaml_file << "both_directions: " << obj_args.both_directions << "\n";
    yaml_file << "scan_point_factor: " << obj_args.scan_point_factor << "\n";
    yaml_file << "smoothness_factor: " << obj_args.smoothness_factor << "\n";
    yaml_file << "smoothness_derivative_type: "
              << static_cast<int>(obj_args.smoothness_derivative_type) << "\n";
    yaml_file << "project_derivative: " << obj_args.project_derivative << "\n";
    yaml_file << "normal_knn: " << obj_args.normal_knn << "\n";
    yaml_file << "normal_search_radius: " << obj_args.normal_search_radius << "\n";
    yaml_file << "odometry_factor: " << obj_args.odometry_factor << "\n";
    yaml_file << "initial_lambda: " << args.optimization_args.initial_lambda << "\n";
    yaml_file << "lambda_factor: " << args.optimization_args.lambda_factor << "\n";
    yaml_file << "max_iters: " << args.optimization_args.max_iters << "\n";
    yaml_file << "tolerance: " << args.optimization_args.tolerance << "\n";
    yaml_file.close();

    // Run the optimization process
    cholmod_common c;
    cholmod_start(&c);
    run_optimization(functor, params, c, point_clouds, args.map_args, scans.frames[0], args,
                     exp_name);
    cholmod_finish(&c);

    // Visualize the optimized map
    visualize_map(params, point_clouds, args.map_args, scans.frames[0], args.vis_args,
                  args.objective_args, /* visualize*/ true, /* save_file */ true, exp_name);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
