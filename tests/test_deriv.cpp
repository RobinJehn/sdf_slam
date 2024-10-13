#include "optimization/utils.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>

// Function to compute the 2D transformation given a point and angle
Eigen::Vector2d transform_2d(const Eigen::Vector2d &p, const double theta) {
  Eigen::Matrix2d rotation;
  rotation << cos(theta), -sin(theta), sin(theta), cos(theta);
  return rotation * p;
}

// Google Test case to test the compute_transformation_derivative_2d function
TEST(TransformationDerivativeTest, CompareWithFiniteDifferences) {
  Eigen::Vector2d p(1.0, 2.0);
  double theta = M_PI / 4;
  double epsilon = 1e-6;

  // Compute analytical derivative
  Eigen::Matrix<double, 2, 3> analytical_derivative =
      compute_transformation_derivative_2d(p, theta);

  // Compute numerical derivatives using finite differences
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

// Main function to run all tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
