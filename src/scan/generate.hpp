#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Function to compute sin(x)
double f(double x) { return std::sin(x); }

// Function to generate range of values and compute sin(x) for each
std::pair<std::vector<double>, std::vector<double>>
f_range(double x_min, double x_max, int num_points) {
  std::vector<double> x(num_points);
  std::vector<double> y(num_points);
  double step = (x_max - x_min) / (num_points - 1);
  for (int i = 0; i < num_points; ++i) {
    x[i] = x_min + i * step;
    y[i] = f(x[i]);
  }
  return {x, y};
}

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
std::pair<double, double> hits_f(double x0, double y0, double theta) {
  auto laser_path = [x0, y0, theta](double x) {
    return y0 + std::tan(theta) * (x - x0);
  };

  auto intersection = [laser_path](double x) { return f(x) - laser_path(x); };

  double search_direction = std::cos(theta) >= 0 ? 1 : -1;
  double step_size = 0.1;
  double step = search_direction * step_size;

  for (int i = 0; i < 100; ++i) {
    double x_a = x0 + i * step;
    double x_b = x0 + (i + 1) * step;
    try {
      double root = bisection(intersection, x_a, x_b);
      return {root, f(root)};
    } catch (const std::exception &) {
      continue;
    }
  }

  return {x0 + 100 * step, laser_path(x0 + 100 * step)};
}

// Function to create a simulated scan of f(x) = sin(x)
std::vector<std::pair<double, double>>
create_scan(double x_scanner, double y_scanner, double theta_scanner,
            double angle_range, int num_points) {
  std::vector<std::pair<double, double>> scan;
  for (int i = 0; i < num_points; ++i) {
    double angle =
        theta_scanner - angle_range / 2 + i * angle_range / (num_points - 1);
    auto [x, y] = hits_f(x_scanner, y_scanner, angle);
    scan.emplace_back(x, y);
  }
  return scan;
}

// Function to create two scans
std::pair<std::vector<std::pair<double, double>>,
          std::vector<std::pair<double, double>>>
create_scans(double x_scanner_1, double y_scanner_1, double theta_scanner_1,
             double x_scanner_2, double y_scanner_2, double theta_scanner_2,
             int num_points = 100, double angle_range = M_PI / 4) {
  auto scan_1 = create_scan(x_scanner_1, y_scanner_1, theta_scanner_1,
                            angle_range, num_points);
  auto scan_2 = create_scan(x_scanner_2, y_scanner_2, theta_scanner_2,
                            angle_range, num_points);
  return {scan_1, scan_2};
}

// Function to print the guess
void print_guess(const Eigen::Vector3d &optimized_theta_tx_ty,
                 const Eigen::Vector3d &true_theta_tx_ty) {
  std::cout << "Optimized parameters:\n" << optimized_theta_tx_ty << "\n";
  std::cout << "True parameters:\n" << true_theta_tx_ty << "\n";
}

#endif // GENERATE_HPP