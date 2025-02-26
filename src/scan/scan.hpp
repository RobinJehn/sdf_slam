#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include "scene.hpp"

/**
 * @brief Create a simulated scan (a point cloud) in the scanner frame.
 *
 * If no shape is hit along a ray, a point at the maximum range is returned.
 *
 * @param scene The scene to scan.
 * @param scanner_position The position of the scanner in 2D space.
 * @param theta_scanner The orientation of the scanner in radians.
 * @param angle_range The total angular range of the scan in radians.
 * @param num_points The number of points to generate in the scan.
 * @param max_range The maximum scanning distance.
 *
 * @return A pointer to a point cloud representing the scan.
 */
pcl::PointCloud<pcl::PointXY>::Ptr create_scan(const Scene &scene,
                                               const Eigen::Vector2d &scanner_position,
                                               double theta_scanner, double angle_range,
                                               int num_points, double max_range = 10.0);

/**
 * @brief Creates a set of scans from the given scene.
 *
 * This function generates a vector of point clouds, each representing a scan
 * from a specific scanner position and orientation within the scene.
 *
 * @param scene The scene to be scanned.
 * @param scanner_positions A vector of 2D positions where the scanners are located.
 * @param thetas A vector of angles (in radians) representing the orientation of each scanner.
 * @param num_points The number of points to be generated in each scan.
 * @param angle_range The range of angles (in radians) to be covered by each scan.
 * @param max_range The maximum scanning distance.
 *
 * @return A vector of point clouds, each representing a scan from a specific scanner position and
 * orientation.
 */
std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> create_scans(
    const Scene &scene, const std::vector<Eigen::Vector2d> &scanner_positions,
    const std::vector<double> &thetas, int num_points, double angle_range, double max_range);
