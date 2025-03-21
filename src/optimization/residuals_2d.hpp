#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <vector>

#include "map/map.hpp"
#include "state/state.hpp"
#include "utils.hpp"

/**
 * @brief Computes the smoothness residuals in a 2D map.
 *
 * This function calculates the smoothness residuals for a given 2D map using a specified smoothness
 * factor. It utilizes a k-d tree for nearest neighbor search and a point cloud of normals for the
 * computation.
 *
 * @param map The 2D map for which the smoothness residuals are to be computed.
 * @param smoothness_factor A factor that influences the smoothness calculation.
 * @param tree_global A pointer to a k-d tree used for nearest neighbor search in the global map.
 * @param normals_global A pointer to a point cloud containing the normals of the global map.
 * @param type The type of derivative to compute
 * @param project_derivative Whether to project the derivative onto the normal of the point.
 *
 * @return A vector of double values representing the computed smoothness residuals.
 */
std::vector<double> compute_smoothness_residual_2d(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const DerivativeType &type,
    const bool project_derivative);

/**
 * @brief Computes the smoothness residuals in 2D using a forward difference method.
 *
 * This function calculates the smoothness residuals for a given 2D map. The smoothness
 * residuals are used to enforce smoothness constraints in the optimization process.
 *
 * @param map The 2D map for which the smoothness residuals are computed.
 * @param smoothness_factor A factor that scales the smoothness residuals.
 * @param tree_global A pointer to a KdTree used for nearest neighbor search in the global frame.
 * @param normals_global A pointer to a point cloud containing the normals in the global frame.
 * @param project_derivative Whether to project the derivative onto the normal of the point.
 *
 * @return A vector of smoothness residuals.
 */
std::vector<double> compute_smoothness_residual_2d_forward(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const bool project_derivative);

/**
 * @brief Computes the smoothness residuals in 2D using a central difference method.
 *
 * This function calculates the smoothness residuals for a given 2D map. The smoothness
 * residuals are used to enforce smoothness constraints in the optimization process.
 *
 * @param map The 2D map for which the smoothness residuals are computed.
 * @param smoothness_factor A factor that scales the smoothness residuals.
 * @param tree_global A pointer to a KdTree used for nearest neighbor search in the global frame.
 * @param normals_global A pointer to a point cloud containing the normals in the global frame.
 * @param project_derivative Whether to project the derivative onto the normal of the point.
 *
 * @return A vector of smoothness residuals.
 */
std::vector<double> compute_smoothness_residual_2d_central(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const bool project_derivative);

/**
 * @brief Computes the smoothness residuals in 2D using the upwind method.
 *
 * This function calculates the smoothness residuals for a given 2D map. The smoothness residuals
 * are used to measure the deviation of the map from a smooth surface. The upwind method is employed
 * to ensure numerical stability.
 *
 * @param map The 2D map for which the smoothness residuals are to be computed.
 * @param smoothness_factor A factor that controls the influence of smoothness in the residual
 * computation.
 * @param tree_global A pointer to a KdTree used for nearest neighbor search in the global point
 * cloud.
 * @param normals_global A pointer to the point cloud containing the normals of the global points.
 * @param project_derivative Whether to project the derivative onto the normal of the point.
 *
 * @return A vector of smoothness residuals.
 */
std::vector<double> compute_smoothness_residual_2d_upwind(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const bool project_derivative);

/**
 * @brief Computes the residuals for a 2D state given a set of point clouds and objective arguments.
 *
 * @param state The 2D state for which residuals are to be computed.
 * @param point_clouds A vector of point clouds, where each point cloud contains points in 2D space.
 * @param objective_args Additional arguments required for the objective function.
 * @return Eigen::VectorXd A vector containing the computed residuals.
 */
Eigen::VectorXd compute_residuals_2d(const State<2> &state,
                                     const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds,
                                     const ObjectiveArgs &objective_args);

/**
 * @brief Computes the roughness residual in 2D.
 *
 * This function calculates the roughness residual based on the provided derivatives.
 *
 * @param derivatives An array of two Map<2> objects representing the derivatives.
 * @return The computed roughness residual as a double.
 */
double compute_roughness_residual_2d(const std::array<Map<2>, 2> &derivatives);
