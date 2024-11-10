#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

template <int Dim> struct ObjectiveFunctorCeres {
  using PointType =
      typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type;
  using Vector = std::conditional_t<Dim == 2, Eigen::Vector2d, Eigen::Vector3d>;

  ObjectiveFunctorCeres(
      const std::array<int, Dim> &num_points, const Vector &min_coords,
      const Vector &max_coords,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const int number_of_points, const bool both_directions,
      const double step_size, const int num_inputs, const int num_outputs,
      const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame);

  bool compute_residuals(const Eigen::VectorXd &x,
                         Eigen::VectorXd &residuals) const;

  void df(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobian) const;

  int num_inputs() const;

  int num_outputs() const;

private:
  const std::array<int, Dim> num_map_points_;
  const Vector min_coords_;
  const Vector max_coords_;
  const std::vector<pcl::PointCloud<PointType>> point_clouds_;
  const int number_of_points_;
  const bool both_directions_;
  const double step_size_;
  const int num_inputs_;
  const int num_outputs_;

  /** The initial frame is fixed */
  const Eigen::Transform<double, Dim, Eigen::Affine> initial_frame_;
};

class ManualCostFunction : public ceres::CostFunction {
public:
  ManualCostFunction(ObjectiveFunctorCeres<2> *functor);

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const override;

private:
  ObjectiveFunctorCeres<2> *functor_;
};
