#include "map/map.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <stdexcept>
#include <vector>

// Function to compute sin(x)
double f(double x) { return std::sin(x); }

double bisection(std::function<double(double)> func, double a, double b,
                 double tol = 1e-6, int max_iter = 100) {
  double fa = func(a);
  double fb = func(b);
  if (fa * fb > 0) {
    throw std::runtime_error("Root is not bracketed");
  }
  for (int i = 0; i < max_iter; ++i) {
    double c = 0.5 * (a + b);
    double fc = func(c);
    if (std::abs(fc) < tol) {
      return c;
    }
    if (fa * fc < 0) {
      b = c;
      fb = fc;
    } else {
      a = c;
      fa = fc;
    }
  }
  throw std::runtime_error("Maximum iterations exceeded");
}

// Function to find intersection points of a line with f(x) = sin(x)
Eigen::Vector2d hits_f(const Eigen::Vector2d &initial_point, double theta) {
  double x0 = initial_point.x();
  double y0 = initial_point.y();

  auto laser_path = [x0, y0, theta](double x) {
    return y0 + std::tan(theta) * (x - x0);
  };

  auto intersection = [laser_path](double x) { return f(x) - laser_path(x); };

  double search_direction = std::cos(theta) >= 0 ? 1 : -1;
  double step_size = 0.1;
  double step = search_direction * step_size;

  const int max_steps = 100;
  for (int i = 0; i < 100; ++i) {
    double x_a = x0 + i * step;
    double x_b = x0 + (i + 1) * step;
    try {
      double root = bisection(intersection, x_a, x_b);
      return Eigen::Vector2d(root, f(root));
    } catch (const std::exception &) {
      continue;
    }
  }

  return Eigen::Vector2d(x0 + max_steps * step,
                         laser_path(x0 + max_steps * step));
}

// Function to create a simulated scan of f(x) = sin(x)
pcl::PointCloud<pcl::PointXY>::Ptr
create_scan(const Eigen::Vector2d &scanner_position, const double theta_scanner,
            const double angle_range, const int num_points) {
  pcl::PointCloud<pcl::PointXY>::Ptr scan(new pcl::PointCloud<pcl::PointXY>);

  scan->points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    double angle =
        theta_scanner - angle_range / 2 + i * angle_range / (num_points - 1);
    Eigen::Vector2d point = hits_f(scanner_position, angle);
    pcl::PointXY pcl_point;
    pcl_point.x = point.x();
    pcl_point.y = point.y();
    scan->points.push_back(pcl_point);
  }

  scan->width = scan->points.size();
  scan->height = 1;

  return scan;
}

// Function to create two scans
std::pair<pcl::PointCloud<pcl::PointXY>::Ptr,
          pcl::PointCloud<pcl::PointXY>::Ptr>
create_scans(const Eigen::Vector2d &scanner_position_1,
             const double theta_scanner_1,
             const Eigen::Vector2d &scanner_position_2,
             const double theta_scanner_2, const int num_points = 100,
             const double angle_range = M_PI / 4) {
  auto scan_1 =
      create_scan(scanner_position_1, theta_scanner_1, angle_range, num_points);
  auto scan_2 =
      create_scan(scanner_position_2, theta_scanner_2, angle_range, num_points);
  return {scan_1, scan_2};
}

double distance_squared(double x, double x0, double y0) {
  return std::pow(x - x0, 2) + std::pow(std::sin(x) - y0, 2);
}

double derivative(double x, double x0, double y0) {
  return 2.0 * (x - x0) + 2.0 * (std::sin(x) - y0) * std::cos(x);
}

double find_closest_point(const double x, const double y,
                          const double x_initial) {
  // We'll use the bisection method to find the root of the derivative
  double x_min = x - M_PI;
  double x_max = x + M_PI;
  double tol = 1e-6;
  int max_iter = 100;

  auto deriv = [x, y](double x_guess) { return derivative(x_guess, x, y); };

  // Search for an interval where the derivative changes sign
  int num_subintervals = 100;
  double dx = (x_max - x_min) / num_subintervals;
  for (int i = 0; i < num_subintervals; ++i) {
    double a = x_min + i * dx;
    double b = a + dx;
    double fa = deriv(a);
    double fb = deriv(b);
    if (fa * fb <= 0) {
      // Root is bracketed in [a, b]
      try {
        double x_optimal = bisection(deriv, a, b, tol, max_iter);
        return x_optimal;
      } catch (const std::exception &) {
        continue;
      }
    }
  }
  // If no root is found, return the initial x value
  return x_initial;
}

Map<2> init_map(const double x_min, const double x_max, const double y_min,
                const double y_max, const int num_x, const int num_y,
                const bool from_ground_truth) {
  const std::array<int, 2> num_points = {num_x, num_y};
  const Eigen::Vector2d min_coords(x_min, y_min);
  const Eigen::Vector2d max_coords(x_max, y_max);
  Map<2> map(num_points, min_coords, max_coords);

  if (from_ground_truth) {
    // Create linearly spaced vectors for x and y
    Eigen::VectorXd x_vals = Eigen::VectorXd::LinSpaced(num_x, x_min, x_max);
    Eigen::VectorXd y_vals = Eigen::VectorXd::LinSpaced(num_y, y_min, y_max);

    // Loop over each grid point
    for (int i = 0; i < num_x; ++i) {
      double x_val = x_vals(i);
      for (int j = 0; j < num_y; ++j) {
        const double y_val = y_vals(j);

        // Find x_optimal that minimizes the distance squared
        const double x_optimal = find_closest_point(x_val, y_val, x_val);
        const double y_optimal = std::sin(x_optimal);

        // Compute the signed distance
        const double dist_sq = distance_squared(x_optimal, x_val, y_val);
        const double dist = std::sqrt(dist_sq);
        const double sign = (y_optimal - y_val) >= 0 ? 1.0 : -1.0;

        map.set_value_at({i, j}, sign * dist);
      }
    }
  }

  return map;
}