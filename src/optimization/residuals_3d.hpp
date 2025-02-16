#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <vector>

#include "map/map.hpp"
#include "state/state.hpp"
#include "utils.hpp"

/**
 * @brief Computes the residuals for a 3D state given a set of point clouds and objective arguments.
 *
 * @param state The 3D state for which residuals are to be computed.
 * @param point_clouds A vector of point clouds, where each point cloud contains points in 3D space.
 * @param objective_args Additional arguments required for the objective function.
 * @return A vector containing the computed residuals.
 */
Eigen::VectorXd compute_residuals_3d(
    const State<3> &state, const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds,
    const ObjectiveArgs &objective_args);

/**
 * @brief Computes the roughness residual in 3D.
 *
 * This function calculates the roughness residual based on the provided derivatives.
 *
 * @param derivatives An array of two Map<3> objects representing the derivatives.
 * @return The computed roughness residual as a double.
 */
double compute_roughness_residual_3d(const std::array<Map<3>, 3> &derivatives);
