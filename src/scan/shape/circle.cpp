#include "circle.hpp"

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
  // Format: "circle center_x center_y radius"
  oss << "circle " << center_.x() << " " << center_.y() << " " << radius_;
  return oss.str();
}

std::shared_ptr<Shape> Circle::from_string(const std::string &str) {
  std::istringstream iss(str);
  std::string type;
  iss >> type;  // should be "circle"
  double cx, cy, radius;
  iss >> cx >> cy >> radius;
  return std::make_shared<Circle>(Eigen::Vector2d(cx, cy), radius);
}

std::shared_ptr<Shape> Circle::from_yaml(const YAML::Node &node) {
  double cx = node["center_x"].as<double>();
  double cy = node["center_y"].as<double>();
  double radius = node["radius"].as<double>();
  return std::make_shared<Circle>(Eigen::Vector2d(cx, cy), radius);
}
