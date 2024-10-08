#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <vector>
#include <functional>

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
