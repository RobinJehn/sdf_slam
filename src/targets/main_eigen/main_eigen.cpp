#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "map/map.hpp"
#include "optimization/objective.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "state/state.hpp"

int main(int argc, char *argv[]) {
  const double x_scanner_1 = 3.5;
  const double y_scanner_1 = -5;
  const double theta_scanner_1 = 9 * M_PI / 16;
  const Eigen::Vector2d scanner_position_1(x_scanner_1, y_scanner_1);

  const double x_scanner_2 = 4;
  const double y_scanner_2 = -5;
  const double theta_scanner_2 = 8 * M_PI / 16;
  const Eigen::Vector2d scanner_position_2(x_scanner_2, y_scanner_2);

  const std::vector<Eigen::Vector2d> scanner_positions = {scanner_position_1, scanner_position_2};
  const std::vector<double> thetas = {theta_scanner_1, theta_scanner_2};
  const std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds =
      create_scans(scanner_positions, thetas);

  // Define the parameters for the optimization
  double x_min = -1;
  double x_max = 2 * M_PI + 1;
  double y_min = -6;
  double y_max = 2;
  int map_size_x = 50;
  int map_size_y = 50;
  bool from_ground_truth = false;

  // Initialize the map
  MapArgs<2> map_args;
  map_args.num_points = {map_size_x, map_size_y};
  map_args.min_coords = {x_min, y_min};
  map_args.max_coords = {x_max, y_max};

  Map<2> map(map_args);

  // Define the initial frame and initial guess for the optimization
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame;
  initial_frame = Eigen::Translation<double, 2>(scanner_position_1) *
                  Eigen::Rotation2D<double>(static_cast<double>(theta_scanner_1));
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_2;
  initial_frame_2 = Eigen::Translation<double, 2>(scanner_position_2) *
                    Eigen::Rotation2D<double>(static_cast<double>(theta_scanner_2));

  // Create the objective functor
  ObjectiveArgs objective_args;
  objective_args.scanline_points = 20;
  objective_args.step_size = 0.1;
  objective_args.both_directions = true;

  const int number_of_scanned_points = point_clouds[0].size() * point_clouds.size();
  const int number_of_residuals = number_of_scanned_points * (objective_args.scanline_points + 1);
  ObjectiveFunctor<2> functor(6 + map_size_x * map_size_y, number_of_residuals, map_args,
                              point_clouds, objective_args, initial_frame);

  // Define the initial parameters for the optimization
  std::vector<Eigen::Transform<double, 2, Eigen::Affine>> transformations = {initial_frame,
                                                                             initial_frame_2};
  Eigen::VectorXd initial_params = flatten<2>(State<2>(map, transformations));
  if (initial_params.size() != functor.inputs()) {
    std::cerr << "Error: Initial parameters size does not match the expected "
                 "size by the functor."
              << std::endl;
    std::cerr << "Expected size: " << functor.inputs() << std::endl;
    std::cerr << "Actual size: " << initial_params.size() << std::endl;
    return -1;
  }

  if (functor.inputs() > functor.values()) {
    std::cerr << "Error: Number of inputs is greater than the number of "
                 "residuals."
              << std::endl;
    return -1;
  }

  // Perform the optimization using Levenberg-Marquardt algorithm
  // Define the callback function for optimization
  auto callback = [](const Eigen::VectorXd &x, int iter, double error) {
    std::cout << "Iteration " << iter << ": Error = " << error << std::endl;
  };
  LevenbergMarquardtWithCallback<ObjectiveFunctor<2>> lm_wrapper(functor, callback);
  lm_wrapper.lm.parameters.factor = 10.0;
  lm_wrapper.lm.parameters.maxfev = 100 * (functor.inputs() + 1);
  lm_wrapper.lm.parameters.ftol = 1e-8;
  lm_wrapper.lm.parameters.xtol = 1e-8;
  lm_wrapper.lm.parameters.gtol = 1e-8;
  lm_wrapper.lm.parameters.epsfcn = 1e-10;

  Eigen::LevenbergMarquardtSpace::Status status = lm_wrapper.minimize(initial_params);

  // Extract the optimized parameters
  Eigen::MatrixXd optimized_map(map_size_x, map_size_y);
  for (int i = 0; i < map_size_x; ++i) {
    for (int j = 0; j < map_size_y; ++j) {
      optimized_map(i, j) = initial_params(i * map_size_y + j);
    }
  }

  Eigen::Vector3d optimized_theta_tx_ty_1 = initial_params.segment<3>(map_size_x * map_size_y);
  Eigen::Vector3d optimized_theta_tx_ty_2 = initial_params.segment<3>(map_size_x * map_size_y + 3);

  // Print the results
  std::cout << "Optimization status: " << status << std::endl;
  std::cout << "Optimized theta, tx, ty for scanner 1: " << optimized_theta_tx_ty_1.transpose()
            << std::endl;
  std::cout << "Optimized theta, tx, ty for scanner 2: " << optimized_theta_tx_ty_2.transpose()
            << std::endl;
  std::cout << "Optimized map:" << std::endl;
  std::cout << optimized_map << std::endl;

  // Normalize the map values to the range [0, 255]
  double min_value = optimized_map.minCoeff();
  double max_value = optimized_map.maxCoeff();
  cv::Mat map_image(map_size_x, map_size_y, CV_8UC1);
  for (int i = 0; i < map_size_x; ++i) {
    for (int j = 0; j < map_size_y; ++j) {
      map_image.at<uchar>(i, j) =
          static_cast<uchar>(255 * (optimized_map(i, j) - min_value) / (max_value - min_value));
    }
  }

  // Display the map using OpenCV
  cv::namedWindow("Optimized Map", cv::WINDOW_AUTOSIZE);
  cv::imshow("Optimized Map", map_image);
  cv::waitKey(0);

  return 0;
}
