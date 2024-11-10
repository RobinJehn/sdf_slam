#pragma once
#include "map/utils.hpp"
#include "optimization/utils.hpp"
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
      const MapArgs<Dim> &map_args,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const ObjectiveArgs &objective_args, const int num_inputs,
      const int num_outputs,
      const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame);

  bool compute_residuals(const Eigen::VectorXd &x,
                         Eigen::VectorXd &residuals) const;

  void df(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobian) const;

  int num_inputs() const;

  int num_outputs() const;

private:
  const MapArgs<Dim> map_args_;
  const ObjectiveArgs objective_args_;
  const std::vector<pcl::PointCloud<PointType>> point_clouds_;
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
