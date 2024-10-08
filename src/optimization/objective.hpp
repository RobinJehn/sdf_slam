#pragma once
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

template <int Dim> class State;

template <int Dim>
float objective(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds);

template <int Dim>
Eigen::VectorXd scan_point_residuals(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds);

template <int Dim>
Eigen::VectorXd scan_line_residuals(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const int number_of_points, const bool both_directions,
    const float step_size);

template <int Dim>
Eigen::VectorXd
objective_vec(const State<Dim> &state,
              const std::vector<pcl::PointCloud<typename std::conditional<
                  Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>> &point_clouds,
              const int number_of_points, const bool both_directions,
              const float step_size);

// Helper function to flatten the state into a vector
template <int Dim> Eigen::VectorXd flatten(const State<Dim> &state);

// Helper function to unflatten the state from a vector
template <int Dim>
State<Dim> unflatten(const Eigen::VectorXd &flattened_state,
                     const std::array<int, Dim> &num_points,
                     const Eigen::Matrix<float, Dim, 1> &min_coords,
                     const Eigen::Matrix<float, Dim, 1> &max_coords);

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

template <int Dim> struct ObjectiveFunctor : Functor<double> {
  using PointType =
      typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type;
  using Vector = std::conditional_t<Dim == 2, Eigen::Vector2f, Eigen::Vector3f>;

  ObjectiveFunctor(const int num_inputs, const int num_outputs,
                   const std::array<int, Dim> &num_points,
                   const Vector &min_coords_, const Vector &max_coords_,
                   const std::vector<pcl::PointCloud<PointType>> &point_clouds,
                   const int number_of_points, const bool both_directions,
                   const float step_size);

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

  const std::vector<pcl::PointCloud<PointType>> point_clouds_;
  const std::array<int, Dim> num_map_points_;

  /** Minimum and maximum values in each dimension */
  const Vector min_coords_;
  const Vector max_coords_;

  /** Parameters for point line residuals */
  const int number_of_points_;
  const bool both_directions_;
  const float step_size_;
};
