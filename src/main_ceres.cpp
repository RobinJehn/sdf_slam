#include "map/map.hpp"
#include "optimization/objective_ceres.hpp" // Use the Ceres-based objective
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "scan/generate.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  // Define scanner positions
  double x_scanner_1 = 3.5;
  double y_scanner_1 = -5;
  double theta_scanner_1 = 9 * M_PI / 16;
  Eigen::Vector2d scanner_position_1(x_scanner_1, y_scanner_1);

  double x_scanner_2 = 4;
  double y_scanner_2 = -5;
  double theta_scanner_2 = 8 * M_PI / 16;
  Eigen::Vector2d scanner_position_2(x_scanner_2, y_scanner_2);

  // Create the scans
  const int num_scan_points = 100;
  auto scans =
      create_scans(scanner_position_1, theta_scanner_1, scanner_position_2,
                   theta_scanner_2, num_scan_points);

  pcl::PointCloud<pcl::PointXY>::Ptr scan1 = scans.first;
  pcl::PointCloud<pcl::PointXY>::Ptr scan2 = scans.second;

  // Define the parameters for the optimization
  double x_min = -1;
  double x_max = 2 * M_PI + 1;
  double y_min = -6;
  double y_max = 2;
  int map_size_x = 50;
  int map_size_y = 50;

  // Initialize the map
  std::array<int, 2> num_points = {map_size_x, map_size_y};
  Eigen::Vector2d min_coords(x_min, y_min);
  Eigen::Vector2d max_coords(x_max, y_max);
  Map<2> map(num_points, min_coords, max_coords);

  // Define the initial frame and initial guess for the optimization
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame =
      Eigen::Translation<double, 2>(scanner_position_1) *
      Eigen::Rotation2D<double>(theta_scanner_1);
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_2 =
      Eigen::Translation<double, 2>(scanner_position_2) *
      Eigen::Rotation2D<double>(theta_scanner_2);

  // Create the point clouds vector
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = {*scan1, *scan2};

  // Set the optimization parameters
  const bool both_directions = true;
  const double step_size = 0.1;
  const int num_line_points = 20;
  const int number_of_scanned_points = scan1->size() + scan2->size();
  const int number_of_residuals =
      number_of_scanned_points * (num_line_points + 1);

  // Use the Ceres ObjectiveFunctor for 2D (with manual Jacobian)
  ObjectiveFunctorCeres<2> *functor = new ObjectiveFunctorCeres<2>(
      num_points, min_coords, max_coords, point_clouds, num_line_points,
      both_directions, step_size);

  // Flatten the state into the parameter vector
  std::vector<Eigen::Transform<double, 2, Eigen::Affine>> transformations = {
      initial_frame, initial_frame_2};
  Eigen::VectorXd initial_params = flatten<2>(State<2>(map, transformations));

  // Check parameter size match
  if (initial_params.size() != functor->num_inputs()) {
    std::cerr << "Error: Initial parameters size does not match the expected "
                 "size by the functor."
              << std::endl;
    return -1;
  }

  // Create the Ceres problem
  ceres::Problem problem;

  // Create the manually-defined cost function
  ceres::CostFunction *cost_function = new ManualCostFunction(functor);

  // Add the residual block with manually computed Jacobian
  problem.AddResidualBlock(cost_function, nullptr, initial_params.data());

  // Set solver options
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 100;
  options.num_threads = 4;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "Final parameters: " << initial_params.transpose() << std::endl;

  // Extract the optimized parameters
  Eigen::MatrixXd optimized_map(map_size_x, map_size_y);
  for (int i = 0; i < map_size_x; ++i) {
    for (int j = 0; j < map_size_y; ++j) {
      optimized_map(i, j) = initial_params(i * map_size_y + j);
    }
  }

  Eigen::Vector3d optimized_theta_tx_ty_1 =
      initial_params.segment<3>(map_size_x * map_size_y);
  Eigen::Vector3d optimized_theta_tx_ty_2 =
      initial_params.segment<3>(map_size_x * map_size_y + 3);

  // Print the results
  std::cout << "Optimized theta, tx, ty for scanner 1: "
            << optimized_theta_tx_ty_1.transpose() << std::endl;
  std::cout << "Optimized theta, tx, ty for scanner 2: "
            << optimized_theta_tx_ty_2.transpose() << std::endl;
  std::cout << "Optimized map:" << std::endl;
  std::cout << optimized_map << std::endl;

  // Normalize the map values to the range [0, 255] for display
  double min_value = optimized_map.minCoeff();
  double max_value = optimized_map.maxCoeff();
  cv::Mat map_image(map_size_x, map_size_y, CV_8UC1);
  for (int i = 0; i < map_size_x; ++i) {
    for (int j = 0; j < map_size_y; ++j) {
      map_image.at<uchar>(i, j) = static_cast<uchar>(
          255 * (optimized_map(i, j) - min_value) / (max_value - min_value));
    }
  }

  // Display the optimized map using OpenCV
  cv::namedWindow("Optimized Map", cv::WINDOW_AUTOSIZE);
  cv::imshow("Optimized Map", map_image);
  cv::waitKey(0);

  return 0;
}
