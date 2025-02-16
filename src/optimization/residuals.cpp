#include "residuals.hpp"

#include "residuals_2d.hpp"
#include "residuals_3d.hpp"

template <int Dim>
Eigen::VectorXd compute_residuals(
    const State<Dim> &state,
    const std::vector<typename pcl::PointCloud<PointType<Dim>>> &point_clouds,
    const ObjectiveArgs &objective_args) {
  if constexpr (Dim == 2) {
    return compute_residuals_2d(state, point_clouds, objective_args);
  } else {
    return compute_residuals_3d(state, point_clouds, objective_args);
  }
}

template Eigen::VectorXd compute_residuals<2>(
    const State<2> &state, const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds,
    const ObjectiveArgs &objective_args);

template Eigen::VectorXd compute_residuals<3>(
    const State<3> &state, const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds,
    const ObjectiveArgs &objective_args);
