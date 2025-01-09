#pragma once
#include "map/map.hpp"
#include "map/utils.hpp"
#include "optimization/utils.hpp"
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
      const int num_inputs, const int num_outputs, const MapArgs<Dim> &map_args,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const ObjectiveArgs &objective_args,
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
  const MapArgs<Dim> map_args_;
  const ObjectiveArgs objective_args_;

  /** The initial frame is fixed */
  const Eigen::Transform<double, Dim, Eigen::Affine> initial_frame_;

  // Helper variables
  static constexpr uint rotation_dim_ = (Dim == 3) ? 3 : 1;
  static constexpr uint n_transformation_params_ = Dim + rotation_dim_;

  /** @brief Computes the state and its derivatives for a given dimension.
   *
   * This function template calculates the state and its derivatives based on
   * the specified dimension (Dim). The computation is tailored to the specific
   * requirements of the optimization process in the SLAM (Simultaneous
   * Localization and Mapping) context.
   */
  std::pair<State<Dim>, std::array<Map<Dim>, Dim>>
  compute_state_and_derivatives(const Eigen::VectorXd &x) const;

  std::vector<Eigen::Triplet<double>>
  compute_jacobian_triplets(const Eigen::VectorXd &x) const;

  void fill_jacobian_triplets(
      std::vector<Eigen::Triplet<double>> &tripletList,
      const int total_map_points, int i,
      const Eigen::Matrix<double, 1, n_transformation_params_>
          &dDF_dTransformation,
      const std::array<typename Map<Dim>::index_t, (1 << Dim)>
          &interpolation_indices,
      const Eigen::VectorXd &interpolation_weights,
      const int transformation_index, const double residual_factor) const;

  void fill_dRoughness_dMap(std::vector<Eigen::Triplet<double>> &tripletList,
                            const std::array<Map<Dim>, Dim> &map_derivatives,
                            const double factor) const;

  /**
   * @brief Compute the partial derivatives of the residual with respect to the
   * map.
   *
   * @param tripletList The list of triplets to store the computed partial
   * derivatives.
   * @param residual_index The index of the residual being computed.
   * @param residual_factor The factor used to weight the residual.
   * @param interpolation_indices An array of indices for the interpolation
   * points.
   * @param interpolation_weights The weights for the interpolation.
   *
   */
  void fill_dMap(std::vector<Eigen::Triplet<double>> &tripletList,
                 const uint residual_index, const double residual_factor,
                 const std::array<typename Map<Dim>::index_t, (1 << Dim)>
                     &interpolation_indices,
                 const Eigen::VectorXd &interpolation_weights) const;

  /**
   * @brief Compute the partial derivatives of the residual with respect to the
   * transformation.
   *
   * @param tripletList The list of triplets to store the computed partial
   * derivatives.
   * @param residual_index The index of the residual being computed.
   * @param residual_factor The factor used to weight the residual.
   * @param offset Where to start storing the computed partial derivatives.
   * @param dTransform The derivative of the residual with respect to the
   * transformation.
   */
  void fill_dTransform(std::vector<Eigen::Triplet<double>> &tripletList,
                       const uint residual_index, const double residual_factor,
                       const uint offset,
                       const Eigen::Matrix<double, 1, n_transformation_params_>
                           &dTransform) const;
};
