#include "optimization/utils.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h> // PCL transform utilities

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
