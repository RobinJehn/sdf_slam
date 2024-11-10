#pragma once

#include "map/map.hpp"
#include "map/utils.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <vector>

// Function to compute sin(x)
double f(double x);

// Function to perform bisection method to find root
double bisection(std::function<double(double)> func, double a, double b,
                 double tol = 1e-6, int max_iter = 100);

// Function to find intersection points of a line with f(x) = sin(x)
Eigen::Vector2d hits_f(const Eigen::Vector2d &initial_point, double theta);

/**
 * @brief Creates a scan of points in a 2D plane.
 *
 * This function generates a point cloud representing a scan from a given
 * scanner position and orientation. The scan covers a specified angular range
 * and contains a specified number of points. The point cloud is in the scanner
 * frame.
 *
 * @param scanner_position The position of the scanner in 2D space.
 * @param theta_scanner The orientation of the scanner in radians.
 * @param angle_range The total angular range of the scan in radians.
 * @param num_points The number of points to generate in the scan.
 *
 * @return A pointer to the generated point cloud.
 */
pcl::PointCloud<pcl::PointXY>::Ptr
create_scan(const Eigen::Vector2d &scanner_position, const double theta_scanner,
            const double angle_range, const int num_points);

/**
 * @brief Create simultated scans
 *
 * @param scanner_position_1
 * @param theta_scanner_1
 * @param scanner_position_2
 * @param theta_scanner_2
 * @param num_points
 * @param angle_range
 *
 * @return Point clouds
 */
std::pair<pcl::PointCloud<pcl::PointXY>::Ptr,
          pcl::PointCloud<pcl::PointXY>::Ptr>
create_scans(const Eigen::Vector2d &scanner_position_1,
             const double theta_scanner_1,
             const Eigen::Vector2d &scanner_position_2,
             const double theta_scanner_2, const int num_points = 100,
             const double angle_range = M_PI / 4);

/**
 * @brief Initializes a map with specified dimensions and resolution.
 *
 * This function creates an Eigen::MatrixXd representing a map, with the
 * specified minimum and maximum coordinates for the x and y axes, and the
 * specified number of cells along each axis. The map can optionally be
 * initialized from ground truth data.
 *
 * @param map_args Arguments for initializing the map.
 * @param from_ground_truth A boolean flag indicating whether to initialize
 *                          the map from ground truth data.
 *
 * @return The initialized map.
 */
Map<2> init_map(const MapArgs<2> &map_args, const bool from_ground_truth);

/**
 * @brief Find the closest point to (x, y) that is on the sin curve
 * (x_op, f(x_op))
 *
 * @param x
 * @param y
 * @param x_initial initial guess for optimal x
 *
 * @return optimal x value
 */
double find_closest_point(const double x, const double y,
                          const double x_initial);

/** @brief Distance between (x0, y0) and (x, f(x)) squared */
double distance_squared(double x, double x0, double y0);

/** @brief Derivative of distance squared */
double derivative(double x, double x0, double y0);