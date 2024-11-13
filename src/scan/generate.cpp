#include "generate.hpp"
#include "map/map.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <stdexcept>
#include <vector>

// Function to compute sin(x)
double f(double x) { return std::sin(x); }

double bisection(std::function<double(double)> func, double a, double b,
                 double tol, int max_iter) {
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

pcl::PointCloud<pcl::PointXY>
create_scan(const Eigen::Vector2d &scanner_position, const double theta_scanner,
            const double angle_range, const int num_points) {
  pcl::PointCloud<pcl::PointXY> scan;

  scan.points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    double angle =
        theta_scanner - angle_range / 2 + i * angle_range / (num_points - 1);
    Eigen::Vector2d point = hits_f(scanner_position, angle);
    pcl::PointXY pcl_point;
    pcl_point.x = point.x();
    pcl_point.y = point.y();
    scan.points.push_back(pcl_point);
  }

  scan.width = scan.points.size();
  scan.height = 1;

  // Transform the scan into the scanner frame
  Eigen::Translation<double, 2> translation(scanner_position.x(),
                                            scanner_position.y());
  Eigen::Rotation2Dd rotation(theta_scanner);
  Eigen::Transform<double, 2, Eigen::Affine> transform = translation * rotation;

  pcl::transformPointCloud(scan, scan,
                           transform.inverse().template cast<float>());

  return scan;
}

// Function to create two scans

std::vector<pcl::PointCloud<pcl::PointXY>>
create_scans(const std::vector<Eigen::Vector2d> &scanner_positions,
             const std::vector<double> &thetas, const int num_points,
             const double angle_range) {

  std::vector<pcl::PointCloud<pcl::PointXY>> scans;
  for (int i = 0; i < scanner_positions.size(); ++i) {
    scans.push_back(
        create_scan(scanner_positions[i], thetas[i], angle_range, num_points));
  }

  return scans;
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

Map<2> init_map(const MapArgs<2> &map_args, const bool from_ground_truth) {
  Map<2> map(map_args);

  if (from_ground_truth) {
    // Create linearly spaced vectors for x and y
    Eigen::VectorXd x_vals = Eigen::VectorXd::LinSpaced(
        map_args.num_points[0], map_args.min_coords[0], map_args.max_coords[0]);
    Eigen::VectorXd y_vals = Eigen::VectorXd::LinSpaced(
        map_args.num_points[1], map_args.min_coords[1], map_args.max_coords[1]);

    // Loop over each grid point
    for (int i = 0; i < map_args.num_points[0]; ++i) {
      double x_val = x_vals(i);
      for (int j = 0; j < map_args.num_points[1]; ++j) {
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