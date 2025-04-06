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
int run_optimization(ObjectiveFunctor<2> &functor, Eigen::VectorXd &params,  //
                     cholmod_common &c,                                      //
                     const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                     const MapArgs<2> &map_args,
                     const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
                     const Args<2> &args,          //
                     const std::string &exp_name,  //
                     int total_iter) {
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

  if (gen_args.save_results && total_iter == 0) {
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
    if (c.status != CHOLMOD_OK) {
      cholmod_free_sparse(&lhs, &c);
      throw std::runtime_error("cholmod_analyze failed.");
    }
    int return_code = cholmod_factorize(lhs, lhs_factorised, &c);
    if (return_code != 1 || c.status != CHOLMOD_OK) {
      cholmod_free_sparse(&lhs, &c);
      throw std::runtime_error("cholmod_factorize failed.");
    }

    // Solve the linear system
    cholmod_dense *cholmod_result = cholmod_solve(CHOLMOD_A, lhs_factorised, rhs, &c);
    if (cholmod_result == NULL || c.status != CHOLMOD_OK) {
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&lhs, &c);
      throw std::runtime_error("cholmod_solve failed.");
    }

    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x, jt_residuals.size() * sizeof(double));
    // Check if any entry in delta is NaN
    if (delta.hasNaN()) {
      throw std::runtime_error("Delta contains NaN values.");
    }
    params += delta;
    auto end_time_jac = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span_jac = end_time_jac - start_time_jac;

    if (gen_args.save_results) {
      // Save the parameters to a file
      std::ofstream params_file(exp_name + "/params_" + std::to_string(++total_iter) + ".txt");
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

  return total_iter;
}

int main() {
  try {
    // Setup and read configuration
    Args<2> args = setup_from_yaml<2>("../config/sdf.yml");
    Scans scans = read_scans(args.general_args.data_path);
    const std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = scans.scans;

    std::vector<Eigen::Transform<double, 2, Eigen::Affine>> odometry;
    for (int i = 0; i < scans.frames.size() - 1; ++i) {
      odometry.push_back(scans.frames[i].inverse() * scans.frames[i + 1]);
    }

    // Initialize the map and set up the optimization parameters
    Map<2> map(args.map_args);
    map.set_value(args.general_args.initial_value);

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

    // Save the parameters to a YAML file
    std::string folder_name =
        args.general_args.experiment_name == ""
            ? "experiment_" + std::to_string(std::chrono::system_clock::to_time_t(
                                  std::chrono::system_clock::now()))
            : args.general_args.experiment_name;

    std::string exp_name = "../experiments/" + folder_name;
    std::filesystem::create_directories(exp_name);
    const ObjectiveArgs &obj_args = args.objective_args;
    std::ofstream yaml_file(exp_name + "/params.yml");
    yaml_file << "scan_line_factor: " << obj_args.scan_line_factor << "\n";
    yaml_file << "scanline_points: " << obj_args.scanline_points << "\n";
    yaml_file << "step_size: " << obj_args.step_size << "\n";
    yaml_file << "scan_point_factor: " << obj_args.scan_point_factor << "\n";
    yaml_file << "smoothness_factor: " << obj_args.smoothness_factor << "\n";
    yaml_file << "smoothness_derivative_type: "
              << static_cast<int>(obj_args.smoothness_derivative_type) << "\n";
    yaml_file << "project_derivative: " << obj_args.project_derivative << "\n";
    yaml_file << "normal_knn: " << obj_args.normal_knn << "\n";
    yaml_file << "odometry_factor: " << obj_args.odometry_factor << "\n";
    yaml_file << "initial_lambda: " << args.optimization_args.initial_lambda << "\n";
    yaml_file << "lambda_factor: " << args.optimization_args.lambda_factor << "\n";
    yaml_file << "num_points: [" << map.get_num_points(0) << ", " << map.get_num_points(1) << "]\n";
    yaml_file << "max_iters: " << args.optimization_args.max_iters << "\n";
    yaml_file << "tolerance: " << args.optimization_args.tolerance << "\n";
    yaml_file << "min_coords: [" << map.get_min_coords().x() << ", " << map.get_min_coords().y()
              << "]\n";
    yaml_file << "max_coords: [" << map.get_max_coords().x() << ", " << map.get_max_coords().y()
              << "]\n";
    yaml_file << "initial_value: " << args.general_args.initial_value << "\n";
    yaml_file << "data_path: " << args.general_args.data_path << "\n";
    yaml_file.close();

    Eigen::VectorXd old_params;
    int total_iter = 0;
    for (int i = 1; i < scans.frames.size(); ++i) {
      std::vector<Eigen::Transform<double, 2, Eigen::Affine>> sub_frames;

      // Add the last frame from sub_frames with odometry to sub_frames
      if (i > 1) {
        State<2> state = unflatten<2>(old_params, scans.frames[0], args.map_args);
        sub_frames = state.transformations_;
        Eigen::Transform<double, 2, Eigen::Affine> new_frame = sub_frames.back() * odometry[i - 1];
        sub_frames.push_back(new_frame);
      } else {
        sub_frames.push_back(scans.frames[0]);
        sub_frames.push_back(scans.frames[1]);
      }

      std::vector<pcl::PointCloud<pcl::PointXY>> sub_point_clouds;
      for (auto it = point_clouds.begin(); it != point_clouds.begin() + i + 1; ++it) {
        sub_point_clouds.push_back(pcl::PointCloud<pcl::PointXY>(*it));
      }
      std::vector<Eigen::Transform<double, 2, Eigen::Affine>> sub_odometry;
      for (auto it = odometry.begin(); it != odometry.begin() + i; ++it) {
        sub_odometry.push_back(Eigen::Transform<double, 2, Eigen::Affine>(*it));
      }
      std::cout << "length: " << sub_odometry.size() << std::endl;

      Eigen::VectorXd params = flatten<2>(State<2>(map, sub_frames));
      if (i > 1) {
        // Copy over the old params
        for (int j = 0; j < old_params.size(); ++j) {
          params[j] = old_params[j];
        }
      }

      // Visualize the initial map
      visualize_map(params, sub_point_clouds, args.map_args, sub_frames[0], args.vis_args,
                    args.objective_args, args.vis_args.initial_visualization,
                    /* save_file */ false);

      // Set up the objective functor for optimization
      int num_points = 0;
      for (const auto &cloud : sub_point_clouds) {
        num_points += cloud.size();
      }

      const uint num_odometry_residuals = sub_odometry.size() * 3;  // 3 per odometry reading

      const int num_residuals = num_points * (args.objective_args.scanline_points + 1) +
                                num_smoothing_residuals + num_odometry_residuals;

      ObjectiveFunctor<2> functor(params.size(), num_residuals, args.map_args, sub_point_clouds,
                                  sub_odometry, args.objective_args, scans.frames[0]);

      // Run the optimization process
      cholmod_common c;
      cholmod_start(&c);
      total_iter = run_optimization(functor, params, c, sub_point_clouds, args.map_args,
                                    scans.frames[0], args, exp_name, total_iter);
      cholmod_finish(&c);

      old_params = params;
      // Visualize the optimized map
      visualize_map(params, sub_point_clouds, args.map_args, scans.frames[0], args.vis_args,
                    args.objective_args, args.vis_args.initial_visualization, /* save_file */ true,
                    exp_name);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
