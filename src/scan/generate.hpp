#pragma once

#include "map/map.hpp"
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

// Function to create a simulated scan of f(x) = sin(x)
pcl::PointCloud<pcl::PointXY>::Ptr
create_scan(const Eigen::Vector2d &scanner_position, const double theta_scanner,
            const double angle_range, const int num_points);

// Function to create two scans
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
 * @param x_min The minimum x-coordinate of the map.
 * @param x_max The maximum x-coordinate of the map.
 * @param y_min The minimum y-coordinate of the map.
 * @param y_max The maximum y-coordinate of the map.
 * @param num_x The number of cells along the x-axis.
 * @param num_y The number of cells along the y-axis.
 * @param from_ground_truth A boolean flag indicating whether to initialize
 *                          the map from ground truth data.
 *
 * @return The initialized map.
 */
Map<2> init_map(const double x_min, const double x_max, const double y_min,
                const double y_max, const int num_x, const int num_y,
                const bool from_ground_truth);

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