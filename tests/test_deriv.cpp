#include "optimization/objective.hpp"
#include "optimization/utils.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

Eigen::Vector2d compute_distance_derivative(const Map<2> &map,
                                            const Eigen::Vector2d &point) {
  auto [interpolation_indices, interpolation_weights] =
      get_interpolation_values<2>(point, map);

  Eigen::Vector2d dDF_dPoint;
  double dx = map.get_d(0);
  double dy = map.get_d(1);

  // Compute partial derivative with respect to x
  double b = interpolation_weights[2] + interpolation_weights[3];
  dDF_dPoint(0) = ((1 - b) * (map.get_value_at(interpolation_indices[1]) -
                              map.get_value_at(interpolation_indices[0])) +
                   b * (map.get_value_at(interpolation_indices[3]) -
                        map.get_value_at(interpolation_indices[2]))) /
                  dx;

  // Compute partial derivative with respect to y
  double a = interpolation_weights[1] + interpolation_weights[3];
  dDF_dPoint(1) = ((1 - a) * (map.get_value_at(interpolation_indices[2]) -
                              map.get_value_at(interpolation_indices[0])) +
                   a * (map.get_value_at(interpolation_indices[3]) -
                        map.get_value_at(interpolation_indices[1]))) /
                  dy;

  return dDF_dPoint;
}

TEST(ObjectiveFunctorTest, DistanceDerivativeTestCustomNumerical) {
  std::array<int, 2> num_points = {10, 10};
  Eigen::Vector2d min_coords(0.0, 0.0);
  Eigen::Vector2d max_coords(10.0, 10.0);
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds;
  int number_of_points = 100;
  bool both_directions = true;
  double step_size = 0.1;
  ObjectiveFunctor<2> functor(2, 3, num_points, min_coords, max_coords,
                              point_clouds, number_of_points, both_directions,
                              step_size);

  Map<2> map(num_points, min_coords, max_coords);

  for (int i = 0; i < num_points[0]; ++i) {
    for (int j = 0; j < num_points[1]; ++j) {
      Eigen::Vector2d point = {min_coords[0] + i * map.get_d(0),
                               min_coords[1] + j * map.get_d(1)};
      map.set_value_at({i, j}, point.norm());
    }
  }

  std::array<Map<2>, 2> derivatives = map.df();

  // Define a point to test
  Eigen::Vector2d point(1.0, 2.0);

  // Step 1: Compute the analytical derivative
  double epsilon = 1e-6;
  Eigen::Vector2d numerical_derivative;

  for (int i = 0; i < 2; ++i) {
    Eigen::Vector2d point_plus = point;
    Eigen::Vector2d point_minus = point;

    // Perturb the i-th component of the point
    point_plus[i] += epsilon;
    point_minus[i] -= epsilon;

    // Compute the distance values at the perturbed points
    double distance_plus = map.value(point_plus);
    double distance_minus = map.value(point_minus);

    // Compute the i-th component of the numerical derivative
    numerical_derivative[i] = (distance_plus - distance_minus) / (2 * epsilon);
  }

  // Step 2: Compute the numerical derivative using finite differences
  Eigen::Vector2d custom_derivative = compute_distance_derivative(map, point);

  // Step 3: Compare the analytical and numerical derivatives
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(numerical_derivative[i], custom_derivative[i], 1e-5)
        << "Mismatch at component " << i;
  }
}

TEST(ObjectiveFunctorTest, DistanceDerivativeTestAnalyticalCustom) {
  std::array<int, 2> num_points = {10, 10};
  Eigen::Vector2d min_coords(0.0, 0.0);
  Eigen::Vector2d max_coords(10.0, 10.0);
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds;
  int number_of_points = 100;
  bool both_directions = true;
  double step_size = 0.1;
  ObjectiveFunctor<2> functor(2, 3, num_points, min_coords, max_coords,
                              point_clouds, number_of_points, both_directions,
                              step_size);

  Map<2> map(num_points, min_coords, max_coords);

  for (int i = 0; i < num_points[0]; ++i) {
    for (int j = 0; j < num_points[1]; ++j) {
      Eigen::Vector2d point = {min_coords[0] + i * map.get_d(0),
                               min_coords[1] + j * map.get_d(1)};
      map.set_value_at({i, j}, point.norm());
    }
  }

  std::array<Map<2>, 2> derivatives = map.df();

  // Define a point to test
  Eigen::Vector2d point(1.0, 2.0);

  // Step 1: Compute the analytical derivative
  Eigen::Vector2d analytical_derivative;
  for (int d = 0; d < 2; ++d) {
    analytical_derivative[d] =
        derivatives[d].in_bounds(point) ? derivatives[d].value(point) : 0;
  }

  // Step 2: Compute the numerical derivative using finite differences
  Eigen::Vector2d custom_derivative = compute_distance_derivative(map, point);

  // Step 3: Compare the analytical and numerical derivatives
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(analytical_derivative[i], custom_derivative[i], 1e-5)
        << "Mismatch at component " << i;
  }
}

TEST(ObjectiveFunctorTest, DistanceDerivativeTestAnalyticalNumerical) {
  std::array<int, 2> num_points = {10, 10};
  Eigen::Vector2d min_coords(0.0, 0.0);
  Eigen::Vector2d max_coords(10.0, 10.0);
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds;
  int number_of_points = 100;
  bool both_directions = true;
  double step_size = 0.1;
  ObjectiveFunctor<2> functor(2, 3, num_points, min_coords, max_coords,
                              point_clouds, number_of_points, both_directions,
                              step_size);

  Map<2> map(num_points, min_coords, max_coords);

  for (int i = 0; i < num_points[0]; ++i) {
    for (int j = 0; j < num_points[1]; ++j) {
      Eigen::Vector2d point = {min_coords[0] + i * map.get_d(0),
                               min_coords[1] + j * map.get_d(1)};
      map.set_value_at({i, j}, point.norm());
    }
  }

  std::array<Map<2>, 2> derivatives = map.df();

  // Define a point to test
  Eigen::Vector2d point(1.0, 2.0);

  // Step 1: Compute the analytical derivative
  Eigen::Vector2d analytical_derivative;
  for (int d = 0; d < 2; ++d) {
    analytical_derivative[d] =
        derivatives[d].in_bounds(point) ? derivatives[d].value(point) : 0;
  }

  // Step 2: Compute the numerical derivative using finite differences
  double epsilon = 1e-6;
  Eigen::Vector2d numerical_derivative;

  for (int i = 0; i < 2; ++i) {
    Eigen::Vector2d point_plus = point;
    Eigen::Vector2d point_minus = point;

    // Perturb the i-th component of the point
    point_plus[i] += epsilon;
    point_minus[i] -= epsilon;

    // Compute the distance values at the perturbed points
    double distance_plus = map.value(point_plus);
    double distance_minus = map.value(point_minus);

    // Compute the i-th component of the numerical derivative
    numerical_derivative[i] = (distance_plus - distance_minus) / (2 * epsilon);
  }

  // Step 3: Compare the analytical and numerical derivatives
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(analytical_derivative[i], numerical_derivative[i], 1e-5)
        << "Mismatch at component " << i;
  }
}

TEST(ObjectiveFunctorTest, ComputeTransformationDerivative2D) {
  std::array<int, 2> num_points = {10, 10};
  Eigen::Vector2d min_coords(0.0, 0.0);
  Eigen::Vector2d max_coords(10.0, 10.0);
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds;
  int number_of_points = 100;
  bool both_directions = true;
  double step_size = 0.1;

  ObjectiveFunctor<2> functor(2, 3, num_points, min_coords, max_coords,
                              point_clouds, number_of_points, both_directions,
                              step_size);
  Eigen::Vector2d point(1.0, 2.0);
  Eigen::Transform<double, 2, Eigen::Affine> transform =
      Eigen::Transform<double, 2, Eigen::Affine>::Identity();
  double theta = M_PI / 4;
  transform.rotate(Eigen::Rotation2Dd(theta));

  Eigen::Matrix<double, 2, 3> derivative =
      functor.compute_transformation_derivative(point, transform);

  // Expected derivative
  Eigen::Matrix<double, 2, 3> expected_derivative =
      compute_transformation_derivative_2d(point, theta);

  // Test: Compare computed and expected derivatives
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(derivative(i, j), expected_derivative(i, j), 1e-6);
    }
  }
}

Eigen::Vector2d transform_2d(const Eigen::Vector2d &p, const double theta) {
  Eigen::Matrix2d rotation;
  rotation << cos(theta), -sin(theta), sin(theta), cos(theta);
  return rotation * p;
}

// Function to compute the 2D transformation using PCL
Eigen::Vector2d transform_2d_with_pcl(const Eigen::Vector2d &p,
                                      const double theta) {
  Eigen::Affine2d transform = Eigen::Affine2d::Identity();
  transform.rotate(Eigen::Rotation2Dd(theta));

  pcl::PointXY point(p.x(), p.y());
  pcl::PointCloud<pcl::PointXY> source_cloud;
  source_cloud.push_back(point);
  pcl::PointCloud<pcl::PointXY> transformed_cloud;
  pcl::transformPointCloud(source_cloud, transformed_cloud,
                           transform.template cast<float>());

  return Eigen::Vector2d(transformed_cloud[0].x, transformed_cloud[0].y);
}

// Google Test case to test the compute_transformation_derivative_2d function
TEST(TransformationDerivativeTest, CompareWithFiniteDifferences) {
  Eigen::Vector2d p(1.0, 2.0);
  double theta = M_PI / 4;
  double epsilon = 1e-6;

  // Compute analytical derivative
  Eigen::Matrix<double, 2, 3> analytical_derivative =
      compute_transformation_derivative_2d(p, theta);

  // Compute numerical derivatives using finite differences and PCL
  // transformation
  Eigen::Matrix<double, 2, 3> numerical_derivative;

  // Translation derivative (should be identity)
  numerical_derivative.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();

  // Numerical derivative w.r.t. theta (finite difference)
  Eigen::Vector2d p_plus = transform_2d_with_pcl(p, theta + epsilon);
  Eigen::Vector2d p_minus = transform_2d_with_pcl(p, theta - epsilon);

  Eigen::Vector2d dp_dtheta = (p_plus - p_minus) / (2 * epsilon);
  numerical_derivative.col(2) = dp_dtheta;

  // Test: Compare analytical and numerical derivatives
  ASSERT_NEAR(analytical_derivative(0, 0), numerical_derivative(0, 0), 1e-6);
  ASSERT_NEAR(analytical_derivative(0, 1), numerical_derivative(0, 1), 1e-6);
  ASSERT_NEAR(analytical_derivative(0, 2), numerical_derivative(0, 2), 1e-6);
  ASSERT_NEAR(analytical_derivative(1, 0), numerical_derivative(1, 0), 1e-6);
  ASSERT_NEAR(analytical_derivative(1, 1), numerical_derivative(1, 1), 1e-6);
  ASSERT_NEAR(analytical_derivative(1, 2), numerical_derivative(1, 2), 1e-6);
}

// Google Test case to test the compute_transformation_derivative_2d function
TEST(TransformationDerivativeTest,
     CompareWithFiniteDifferencesManualTransform) {
  Eigen::Vector2d p(1.0, 2.0);
  double theta = M_PI / 4;
  double epsilon = 1e-6;

  // Compute analytical derivative
  Eigen::Matrix<double, 2, 3> analytical_derivative =
      compute_transformation_derivative_2d(p, theta);

  // Compute numerical derivatives using finite differences and PCL
  // transformation
  Eigen::Matrix<double, 2, 3> numerical_derivative;

  // Translation derivative (should be identity)
  numerical_derivative.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();

  // Numerical derivative w.r.t. theta (finite difference)
  Eigen::Vector2d p_plus = transform_2d(p, theta + epsilon);
  Eigen::Vector2d p_minus = transform_2d(p, theta - epsilon);

  Eigen::Vector2d dp_dtheta = (p_plus - p_minus) / (2 * epsilon);
  numerical_derivative.col(2) = dp_dtheta;

  // Test: Compare analytical and numerical derivatives
  ASSERT_NEAR(analytical_derivative(0, 0), numerical_derivative(0, 0), 1e-6);
  ASSERT_NEAR(analytical_derivative(0, 1), numerical_derivative(0, 1), 1e-6);
  ASSERT_NEAR(analytical_derivative(0, 2), numerical_derivative(0, 2), 1e-6);
  ASSERT_NEAR(analytical_derivative(1, 0), numerical_derivative(1, 0), 1e-6);
  ASSERT_NEAR(analytical_derivative(1, 1), numerical_derivative(1, 1), 1e-6);
  ASSERT_NEAR(analytical_derivative(1, 2), numerical_derivative(1, 2), 1e-6);
}

// Additional test cases
TEST(TransformationTest, ZeroRotation) {
  Eigen::Vector2d p(1.0, 2.0);
  double theta = 0.0;

  Eigen::Vector2d result = transform_2d(p, theta);
  Eigen::Vector2d result_pcl = transform_2d_with_pcl(p, theta);

  ASSERT_NEAR(result.x(), result_pcl.x(), 1e-6);
  ASSERT_NEAR(result.y(), result_pcl.y(), 1e-6);
}

TEST(TransformationTest, NinetyDegreeRotation) {
  Eigen::Vector2d p(1.0, 0.0);
  double theta = M_PI / 2;

  Eigen::Vector2d result = transform_2d(p, theta);
  Eigen::Vector2d result_pcl = transform_2d_with_pcl(p, theta);

  ASSERT_NEAR(result.x(), result_pcl.x(), 1e-6);
  ASSERT_NEAR(result.y(), result_pcl.y(), 1e-6);
}

TEST(TransformationTest, NegativeRotation) {
  Eigen::Vector2d p(1.0, 0.0);
  double theta = -M_PI / 2;

  Eigen::Vector2d result = transform_2d(p, theta);
  Eigen::Vector2d result_pcl = transform_2d_with_pcl(p, theta);

  ASSERT_NEAR(result.x(), result_pcl.x(), 1e-6);
  ASSERT_NEAR(result.y(), result_pcl.y(), 1e-6);
}

// Main function to run all tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
