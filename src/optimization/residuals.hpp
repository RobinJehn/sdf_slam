#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "state/state.hpp"
#include "utils.hpp"

template <int Dim>
Eigen::VectorXd compute_residuals(
    const State<Dim> &state,
    const std::vector<typename pcl::PointCloud<PointType<Dim>>> &point_clouds,
    const std::vector<Eigen::Transform<double, Dim, Eigen::Affine>> &odometry,
    const ObjectiveArgs &objective_args);
