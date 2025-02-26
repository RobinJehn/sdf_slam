#include "shape.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

bool Sinusoid::intersect_ray(const Eigen::Vector2d &origin, double angle,
                             Eigen::Vector2d &intersection, double &t) const {
  double x0 = origin.x();
  double y0 = origin.y();
  // For a given x, the ray’s y coordinate is:
  double tan_angle = std::tan(angle);
  auto laser_path = [x0, y0, tan_angle](double x) { return y0 + tan_angle * (x - x0); };
  // We need to solve: sin(x) = laser_path(x)
  auto func = [laser_path](double x) { return std::sin(x) - laser_path(x); };

  // Decide on the search direction: if cos(angle) is positive, x increases with t.
  double search_direction = std::cos(angle) >= 0 ? 1.0 : -1.0;
  double step_size = 0.1;
  double step = search_direction * step_size;

  const int max_steps = 100;
  double root = 0;
  bool found = false;
  for (int i = 0; i < max_steps; ++i) {
    double x_a = x0 + i * step;
    double x_b = x0 + (i + 1) * step;
    double fa = func(x_a);
    double fb = func(x_b);
    if (fa * fb <= 0) {  // A sign change means a root is bracketed.
      // Bisection method:
      double a = x_a, b = x_b;
      const double tol = 1e-6;
      const int max_iter = 100;
      for (int iter = 0; iter < max_iter; ++iter) {
        double c = 0.5 * (a + b);
        double fc = func(c);
        if (std::abs(fc) < tol) {
          root = c;
          found = true;
          break;
        }
        if (fa * fc < 0) {
          b = c;
          fb = fc;
        } else {
          a = c;
          fa = fc;
        }
      }
      if (found) break;
    }
  }
  if (!found) {
    // No intersection was found.
    return false;
  }
  // Intersection point on the sinusoid:
  intersection = Eigen::Vector2d(root, std::sin(root));
  // Compute t from the x component: (x - x0) = t*cos(angle)
  t = (root - x0) / std::cos(angle);
  if (t < 0) return false;  // Intersection lies behind the scanner.
  return true;
}

double Sinusoid::distance(const Eigen::Vector2d &p) const {
  // p = (x0, y0). We want to minimize:
  //   D(x) = sqrt((x - x0)^2 + (sin(x) - y0)^2)
  // For simplicity, sample candidate x–values near x0.
  double x0 = p.x();
  const double delta = 0.001;
  const int num_samples = 1000;
  double best_d2 = std::numeric_limits<double>::max();
  double best_x = x0;
  // Sample in an interval around x0.
  for (int i = -num_samples / 2; i <= num_samples / 2; ++i) {
    double x_candidate = x0 + i * delta;
    double d2 = std::pow(x_candidate - x0, 2) + std::pow(std::sin(x_candidate) - p.y(), 2);
    if (d2 < best_d2) {
      best_d2 = d2;
      best_x = x_candidate;
    }
  }
  // Here, we consider the distance positive (i.e. we don’t define an “inside” for the sinusoid).
  return (p.y() > std::sin(best_x)) ? -std::sqrt(best_d2) : std::sqrt(best_d2);
}

std::string Sinusoid::to_string() const {
  // Sinusoid has no parameters so we simply output its type.
  return "Sinusoid";
}

std::shared_ptr<Shape> Sinusoid::from_string(const std::string &str) {
  // You could parse the string if needed; here we simply return a new Sinusoid.
  return std::make_shared<Sinusoid>();
}

Circle::Circle(const Eigen::Vector2d &center, double radius) : center_(center), radius_(radius) {}

bool Circle::intersect_ray(const Eigen::Vector2d &origin, double angle,
                           Eigen::Vector2d &intersection, double &t) const {
  Eigen::Vector2d dir(std::cos(angle), std::sin(angle));
  // The ray: origin + t * dir, t >= 0.
  // Solve for t such that ||origin + t*dir - center|| = radius.
  Eigen::Vector2d oc = origin - center_;
  double A = dir.dot(dir);  // = 1.0
  double B = 2 * oc.dot(dir);
  double C = oc.dot(oc) - radius_ * radius_;
  double discriminant = B * B - 4 * A * C;
  if (discriminant < 0) return false;  // No intersection.
  double sqrt_disc = std::sqrt(discriminant);
  double t1 = (-B - sqrt_disc) / (2 * A);
  double t2 = (-B + sqrt_disc) / (2 * A);
  // Choose the smallest positive t.
  t = std::numeric_limits<double>::max();
  if (t1 > 1e-6 && t1 < t) t = t1;
  if (t2 > 1e-6 && t2 < t) t = t2;
  if (t == std::numeric_limits<double>::max()) return false;
  intersection = origin + t * dir;
  return true;
}

double Circle::distance(const Eigen::Vector2d &p) const {
  // Euclidean distance from p to the circle's boundary.
  // Negative inside the circle, positive outside.
  return (p - center_).norm() - radius_;
}

std::string Circle::to_string() const {
  std::ostringstream oss;
  // Format: "Circle center_x center_y radius"
  oss << "Circle " << center_.x() << " " << center_.y() << " " << radius_;
  return oss.str();
}

std::shared_ptr<Shape> Circle::from_string(const std::string &str) {
  std::istringstream iss(str);
  std::string type;
  iss >> type;  // should be "Circle"
  double cx, cy, radius;
  iss >> cx >> cy >> radius;
  return std::make_shared<Circle>(Eigen::Vector2d(cx, cy), radius);
}
