#include "map/map.hpp"
#include "optimization/objective.hpp"
#include "scan/generate.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <iostream>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

int main(int argc, char *argv[]) {
  double x_scanner_1 = 3.5;
  double y_scanner_1 = -5;
  double theta_scanner_1 = 9 * M_PI / 16;
  Eigen::Vector2d scanner_position_1(x_scanner_1, y_scanner_1);

  double x_scanner_2 = 4;
  double y_scanner_2 = -5;
  double theta_scanner_2 = 8 * M_PI / 16;
  Eigen::Vector2d scanner_position_2(x_scanner_2, y_scanner_2);

  auto scans = create_scans(scanner_position_1, theta_scanner_1,
                            scanner_position_2, theta_scanner_2);

  pcl::PointCloud<pcl::PointXY>::Ptr scan1 = scans.first;
  pcl::PointCloud<pcl::PointXY>::Ptr scan2 = scans.second;

  std::cout << "Scan 1 points:" << std::endl;
  for (const auto &point : scan1->points) {
    std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
  }

  std::cout << "Scan 2 points:" << std::endl;
  for (const auto &point : scan2->points) {
    std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
  }
  // Define the parameters for the optimization
  double x_min = -1;
  double x_max = 2 * M_PI + 1;
  double y_min = -6;
  double y_max = 2;
  int map_size_x = 50;
  int map_size_y = 50;
  bool from_ground_truth = false;

  // Initialize the map
  std::array<int, 2> num_points = {map_size_x, map_size_y};
  Eigen::Vector2f min_coords(x_min, y_min);
  Eigen::Vector2f max_coords(x_max, y_max);
  Map<2> map(num_points, min_coords, max_coords);

  // Define the initial frame and initial guess for the optimization
  Eigen::Transform<float, 2, Eigen::Affine> initial_frame;
  initial_frame =
      Eigen::Translation<float, 2>(scanner_position_1.cast<float>()) *
      Eigen::Rotation2D<float>(static_cast<float>(theta_scanner_1));
  Eigen::Transform<float, 2, Eigen::Affine> initial_frame_2;
  initial_frame_2 =
      Eigen::Translation<float, 2>(scanner_position_2.cast<float>()) *
      Eigen::Rotation2D<float>(static_cast<float>(theta_scanner_2));

  // Create the point clouds vector
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = {*scan1, *scan2};

  // Create the objective functor
  ObjectiveFunctor<2> functor(
      6 + map_size_x * map_size_y, map_size_x * map_size_y, num_points,
      min_coords.cast<float>(), max_coords.cast<float>(), point_clouds);

  // Define the initial parameters for the optimization
  std::vector<Eigen::Transform<float, 2, Eigen::Affine>> transformations = {
      initial_frame, initial_frame_2};
  Eigen::VectorXd initial_params = flatten<2>(State<2>(map, transformations));

  // Perform the optimization using Levenberg-Marquardt algorithm
  Eigen::NumericalDiff<ObjectiveFunctor<2>> num_diff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ObjectiveFunctor<2>>> lm(
      num_diff);
  Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(initial_params);

  // Extract the optimized parameters
  Eigen::Vector3d optimized_theta_tx_ty_1 = initial_params.segment<3>(0);
  Eigen::Vector3d optimized_theta_tx_ty_2 = initial_params.segment<3>(3);
  Eigen::MatrixXd optimized_map(map_size_x, map_size_y);
  for (int i = 0; i < map_size_x; ++i) {
    for (int j = 0; j < map_size_y; ++j) {
      optimized_map(i, j) = initial_params(6 + i * map_size_y + j);
    }
  }

  // Print the results
  std::cout << "Optimization status: " << status << std::endl;
  std::cout << "Optimized theta, tx, ty for scanner 1: "
            << optimized_theta_tx_ty_1.transpose() << std::endl;
  std::cout << "Optimized theta, tx, ty for scanner 2: "
            << optimized_theta_tx_ty_2.transpose() << std::endl;
  std::cout << "Optimized map:" << std::endl;
  std::cout << optimized_map << std::endl;

  return 0;
}