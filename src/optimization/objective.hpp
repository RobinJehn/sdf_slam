#pragma once
#include "map/map.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

template <int Dim> class State;

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
  using Vector = std::conditional_t<Dim == 2, Eigen::Vector2d, Eigen::Vector3d>;

  ObjectiveFunctor(
      const int num_inputs, const int num_outputs,
      const std::array<int, Dim> &num_points, const Vector &min_coords_,
      const Vector &max_coords_,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const int number_of_points, const bool both_directions,
      const double step_size,
      const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame);

  /**
   * @brief Calculate the objective function (residuals) for a given input
   * vector.
   *
   * @param x The map and transformation parameters (excluding the first
   * transformation that is fixed).
   * @param fvec Residuals of the objective function.
   * @return Returns 0 to indicate success.
   */
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

  /** @brief Computes the sparse Jacobian matrix for a given input vector.
   *
   * @param x The input vector for which the Jacobian is to be computed.
   * @param jacobian The sparse matrix where the computed Jacobian will be
   * stored.
   * @return An integer indicating the success or failure of the computation.
   */
  int sparse_df(const Eigen::VectorXd &x,
                Eigen::SparseMatrix<double> &jacobian) const;

  /**
   * @brief Compute the partial derivatives of the objective function
   * (residuals) with respect to each parameter in `x`.
   *
   * @param x The current parameter values
   * @param jacobian The matrix to store the computed partial derivatives
   * @return int 0 to indicate success
   */
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobian) const;

private:
  const std::vector<pcl::PointCloud<PointType>> point_clouds_;
  const std::array<int, Dim> num_map_points_;

  /** Minimum and maximum values in each dimension */
  const Vector min_coords_;
  const Vector max_coords_;

  /** Parameters for point line residuals */
  const int number_of_points_;
  const bool both_directions_;
  const double step_size_;

  /** The initial frame is fixed */
  const Eigen::Transform<double, Dim, Eigen::Affine> initial_frame_;

  /** @brief Computes the state and its derivatives for a given dimension.
   *
   * This function template calculates the state and its derivatives based on
   * the specified dimension (Dim). The computation is tailored to the specific
   * requirements of the optimization process in the SLAM (Simultaneous
   * Localization and Mapping) context.
   */
  std::pair<State<Dim>, std::array<Map<Dim>, Dim>>
  compute_state_and_derivatives(
      const Eigen::VectorXd &x, const std::array<int, Dim> &num_map_points,
      const Eigen::Matrix<double, Dim, 1> &min_coords,
      const Eigen::Matrix<double, Dim, 1> &max_coords) const;

  /**
   * @brief Fills the dense Jacobian matrix for the given transformation.
   *
   * This function computes and fills the Jacobian matrix for a specific
   * transformation based on the provided parameters. It uses the derivatives of
   * the distance function with respect to the point and the point with respect
   * to the transformation to populate the Jacobian matrix.
   *
   * @param jacobian The matrix to be filled with the Jacobian values.
   * @param num_map_points An array representing the number of map points in
   * each dimension.
   * @param i The index of the current point being processed.
   * @param dDF_dPoint The derivative of the distance function with respect to
   * the point.
   * @param dPoint_dTransformation The derivative of the point with respect to
   * the transformation.
   * @param point_clouds A vector of point clouds containing the points to be
   * processed.
   * @param interpolation_point_indices An array of indices for the
   * interpolation points.
   * @param interpolation_weights The weights for the interpolation.
   * @param transformation_index The index of the transformation being applied.
   */
  void fill_jacobian_dense(
      Eigen::MatrixXd &jacobian, const std::array<int, Dim> &num_map_points,
      int i, const Eigen::Matrix<double, 1, Dim> &dDF_dPoint,
      const Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)>
          &dPoint_dTransformation,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const std::array<typename Map<Dim>::index_t, (1 << Dim)>
          &interpolation_point_indices,
      const Eigen::VectorXd &interpolation_weights,
      const int transformation_index) const;

  /**
   * @brief Fills a sparse Jacobian matrix with the provided data.
   *
   * This function populates a sparse Jacobian matrix using the provided
   * parameters, which include the number of map points, derivatives of the
   * distance function with respect to the point, derivatives of the point with
   * respect to the transformation, point clouds, interpolation point indices,
   * interpolation weights, and the transformation index.
   *
   * @param jacobian A reference to a vector of Eigen::Triplet<double> to store
   * the sparse Jacobian entries.
   * @param num_map_points An array representing the number of map points in
   * each dimension.
   * @param i An integer index representing the current point.
   * @param dDF_dPoint A matrix representing the derivative of the distance
   * function with respect to the point.
   * @param dPoint_dTransformation A matrix representing the derivative of the
   * point with respect to the transformation.
   * @param point_clouds A vector of point clouds containing the points to be
   * used in the Jacobian computation.
   * @param interpolation_point_indices An array of indices representing the
   * interpolation points.
   * @param interpolation_weights A vector of weights for the interpolation
   * points.
   * @param transformation_index An integer index representing the current
   * transformation.
   */
  void fill_jacobian_sparse(
      std::vector<Eigen::Triplet<double>> &jacobian,
      const std::array<int, Dim> &num_map_points, int i,
      const Eigen::Matrix<double, 1, Dim> &dDF_dPoint,
      const Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)>
          &dPoint_dTransformation,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const std::array<typename Map<Dim>::index_t, (1 << Dim)>
          &interpolation_point_indices,
      const Eigen::VectorXd &interpolation_weights,
      const int transformation_index) const;
};
