#include "map/map.hpp"
#include "map/utils.hpp"
#include "optimization/objective_ceres.hpp"
#include "optimization/utils.hpp"
#include "scan/generate.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// Helper function to plot a map with sine function overlay
void plot_map(const Map<2> &map, int map_size_x, int map_size_y, double x_min,
              double x_max, double y_min, double y_max,
              const std::string &window_name) {
  const double max_val_all = 5.0;
  const double min_val_all = -5.0;

  cv::Mat map_image(map_size_x, map_size_y, CV_8UC1);
  const double min_val = std::max(map.get_min_value(), min_val_all);
  const double max_val = std::min(map.get_max_value(), max_val_all);
  for (int i = 0; i < map_size_x; ++i) {
    for (int j = 0; j < map_size_y; ++j) {
      double value = map.get_value_at({i, j});
      value = std::clamp(value, min_val_all, max_val_all);
      map_image.at<uchar>(map_size_y - j, i) =
          static_cast<uchar>(255 * (value - min_val) / (max_val - min_val));
    }
  }

  // Overlay sine function
  for (int i = 0; i < map_size_x; ++i) {
    double x = x_min + i * (x_max - x_min) / map_size_x;
    double y = std::sin(x);
    int j = static_cast<int>((y - y_min) / (y_max - y_min) * map_size_y);
    if (j >= 0 && j < map_size_y) {
      map_image.at<uchar>(map_size_y - j, i) = 255;
    }
  }

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name, map_image);
  cv::waitKey(0);
}

// Helper function to output optimization results to file
void save_summary_to_file(const std::string &filename,
                          const ceres::Solver::Summary &summary) {
  std::ofstream summary_file(filename);
  if (summary_file.is_open()) {
    summary_file << summary.FullReport() << "\n";
    summary_file.close();
  } else {
    std::cerr << "Unable to open file for writing summary." << std::endl;
  }
}

// Main function
int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  // Define scanner positions and orientations
  Eigen::Vector2d scanner_position_1(3.5, -5);
  Eigen::Vector2d scanner_position_2(4, -5);
  double theta_scanner_1 = 9 * M_PI / 16;
  double theta_scanner_2 = 8 * M_PI / 16;

  // Create the scans
  constexpr int num_scan_points = 100;
  auto scans =
      create_scans(scanner_position_1, theta_scanner_1, scanner_position_2,
                   theta_scanner_2, num_scan_points);
  pcl::PointCloud<pcl::PointXY>::Ptr scan1 = scans.first;
  pcl::PointCloud<pcl::PointXY>::Ptr scan2 = scans.second;

  // Define map parameters
  constexpr double x_min = -3;
  constexpr double x_max = 2 * M_PI + 3;
  constexpr double y_min = -8;
  constexpr double y_max = 4;
  constexpr int map_size_x = 50;
  constexpr int map_size_y = 50;

  MapArgs<2> map_args;
  map_args.num_points = {map_size_x, map_size_y};
  map_args.min_coords = Eigen::Vector2d(x_min, y_min);
  map_args.max_coords = Eigen::Vector2d(x_max, y_max);

  // Initialize the map
  Map<2> map = init_map(map_args, true);

  // Plot the initial map
  plot_map(map, map_size_x, map_size_y, x_min, x_max, y_min, y_max,
           "Initial Map");

  // Set up initial transformations
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_1 =
      Eigen::Translation<double, 2>(scanner_position_1) *
      Eigen::Rotation2D<double>(theta_scanner_1);
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_2 =
      Eigen::Translation<double, 2>(scanner_position_2) *
      Eigen::Rotation2D<double>(theta_scanner_2);

  // Create the point clouds vector
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = {*scan1, *scan2};

  // Set up optimization parameters
  const bool both_directions = true;
  constexpr int num_line_points = 20;
  const double step_size = 0.1;
  constexpr int number_of_scanned_points = 2 * num_scan_points;
  constexpr int num_residuals =
      number_of_scanned_points * (num_line_points + 1);
  const int num_parameters =
      (point_clouds.size() - 1) * 3 + map_size_x * map_size_y;

  // Objective Functor for 2D with Ceres
  ObjectiveFunctorCeres<2> *functor = new ObjectiveFunctorCeres<2>(
      map_args, point_clouds, num_line_points, both_directions, step_size,
      num_parameters, num_residuals, initial_frame_1);

  // Flatten the state into the parameter vector
  std::vector<Eigen::Transform<double, 2, Eigen::Affine>> transformations = {
      initial_frame_1, initial_frame_2};
  Eigen::VectorXd params = flatten<2>(State<2>(map, transformations));

  // Ensure parameter size consistency
  if (params.size() != functor->num_inputs()) {
    std::cerr << "Error: Initial parameters size mismatch." << std::endl;
    return -1;
  }

  // Create the Ceres problem
  ceres::Problem problem;
  ceres::CostFunction *cost_function = new ManualCostFunction(functor);
  problem.AddResidualBlock(cost_function, nullptr, params.data());

  // Set solver options
  ceres::Solver::Options options;
  options.check_gradients = true;
  // options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
  options.gradient_check_relative_precision = 1e-4;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 10;
  options.num_threads = 4;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Save summary to file
  save_summary_to_file("optimization_summary.txt", summary);

  // Unflatten the optimized parameters back into the state
  State<2> optimized_state = unflatten<2>(params, initial_frame_1, map_args);

  plot_map(optimized_state.map_, map_size_x, map_size_y, x_min, x_max, y_min,
           y_max, "Optimized Map");

  return 0;
}
