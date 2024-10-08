#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

class State;

float objective(
    const State &state,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds);

Eigen::VectorXd
objective_vec(const State &state,
              const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds);

// Helper function to flatten the state into a vector
Eigen::VectorXd flatten(const State &state);

// Helper function to unflatten the state from a vector
State unflatten(const Eigen::VectorXd &flattened_state,
                const int num_map_points, const float min_x, const float max_x,
                const float min_y, const float max_y, const float min_z,
                const float max_z);

// Generic functor
template <typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor {
  typedef _Scalar Scalar;
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>
      JacobianType;

  int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }
};

struct ObjectiveFunctor : Functor<double> {
  ObjectiveFunctor(
      const int num_inputs, const int num_outputs, const int num_map_points,
      const float min_x, const float max_x, const float min_y,
      const float max_y, const float min_z, const float max_z,
      const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds);

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

  const std::vector<pcl::PointCloud<pcl::PointXYZ>> point_clouds_;
  const int num_map_points_;
  const float min_x_;
  const float max_x_;
  const float min_y_;
  const float max_y_;
  const float min_z_;
  const float max_z_;
};
