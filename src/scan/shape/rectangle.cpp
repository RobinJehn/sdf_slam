#include "rectangle.hpp"

#include <cmath>
#include <limits>
#include <sstream>

// Constructor
Rectangle::Rectangle(const Eigen::Vector2d &center, double width, double height, double orientation)
    : center_(center), width_(width), height_(height), orientation_(orientation) {}

// intersect_ray: transform the ray into the rectangle's local frame (where it is axis aligned),
// then apply the slab method.
bool Rectangle::intersect_ray(const Eigen::Vector2d &origin, double angle,
                              Eigen::Vector2d &intersection, double &t) const {
  // Compute the ray's direction in global frame.
  Eigen::Vector2d dir(std::cos(angle), std::sin(angle));

  // Create a rotation matrix to transform points from global to rectangle-local coordinates.
  // We need to rotate by -orientation_.
  double cos_theta = std::cos(orientation_);
  double sin_theta = std::sin(orientation_);
  Eigen::Matrix2d R;
  R << cos_theta, sin_theta, -sin_theta, cos_theta;

  // Transform ray origin and direction into local frame.
  Eigen::Vector2d local_origin = R * (origin - center_);
  Eigen::Vector2d local_dir = R * dir;

  // Define the axis-aligned bounds in local coordinates.
  double x_min = -width_ / 2.0, x_max = width_ / 2.0;
  double y_min = -height_ / 2.0, y_max = height_ / 2.0;

  // Initialize tmin and tmax for the slab method.
  double tmin = -std::numeric_limits<double>::infinity();
  double tmax = std::numeric_limits<double>::infinity();
  const double epsilon = 1e-8;

  // Process each axis (0 for x, 1 for y).
  for (int i = 0; i < 2; ++i) {
    double ro = local_origin[i];
    double rd = local_dir[i];
    double slab_min = (i == 0) ? x_min : y_min;
    double slab_max = (i == 0) ? x_max : y_max;

    if (std::abs(rd) < epsilon) {
      // The ray is nearly parallel to this slab. If the origin is not within the slab, no hit.
      if (ro < slab_min || ro > slab_max) {
        return false;
      }
      // Otherwise, this axis does not constrain t.
    } else {
      double t1 = (slab_min - ro) / rd;
      double t2 = (slab_max - ro) / rd;
      if (t1 > t2) std::swap(t1, t2);
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return false;
      if (tmax < 1e-6) return false;  // Intersection is behind the ray.
    }
  }

  // Choose t: if tmin is positive use it; otherwise (ray starts inside) use tmax.
  t = (tmin > 1e-6) ? tmin : tmax;
  if (t < 1e-6) return false;  // Still behind the ray origin.

  // Compute the intersection point in local coordinates.
  Eigen::Vector2d local_intersection = local_origin + t * local_dir;

  // Transform back to global coordinates.
  // The inverse rotation (from local back to global) is the transpose of R.
  Eigen::Matrix2d R_inv = R.transpose();
  intersection = center_ + R_inv * local_intersection;
  return true;
}

// distance: compute the signed distance from point p to the rectangle.
// Transform p into the rectangle's local frame.
double Rectangle::distance(const Eigen::Vector2d &p) const {
  double cos_theta = std::cos(orientation_);
  double sin_theta = std::sin(orientation_);
  Eigen::Matrix2d R;
  R << cos_theta, sin_theta, -sin_theta, cos_theta;
  Eigen::Vector2d local_p = R * (p - center_);

  // Compute the difference between the absolute local coordinates and half-dimensions.
  double dx = std::abs(local_p.x()) - width_ / 2.0;
  double dy = std::abs(local_p.y()) - height_ / 2.0;

  if (dx > 0.0 || dy > 0.0) {
    // Outside the rectangle: Euclidean distance to the boundary.
    double outside_dx = (dx > 0.0) ? dx : 0.0;
    double outside_dy = (dy > 0.0) ? dy : 0.0;
    return std::sqrt(outside_dx * outside_dx + outside_dy * outside_dy);
  } else {
    // Inside the rectangle: negative distance (smallest distance to an edge).
    double dist_to_edge =
        std::min(width_ / 2.0 - std::abs(local_p.x()), height_ / 2.0 - std::abs(local_p.y()));
    return -dist_to_edge;
  }
}

// to_string: produce a string representation.
// Format: "rectangle center_x center_y width height orientation"
std::string Rectangle::to_string() const {
  std::ostringstream oss;
  oss << "rectangle " << center_.x() << " " << center_.y() << " " << width_ << " " << height_ << " "
      << orientation_;
  return oss.str();
}

// from_string: parse a string to create a Rectangle instance.
std::shared_ptr<Shape> Rectangle::from_string(const std::string &str) {
  std::istringstream iss(str);
  std::string type;
  iss >> type;  // should be "rectangle"
  double cx, cy, width, height, orientation;
  iss >> cx >> cy >> width >> height >> orientation;
  return std::make_shared<Rectangle>(Eigen::Vector2d(cx, cy), width, height, orientation);
}

// from_yaml: create a Rectangle from a YAML node.
// Expected keys: center_x, center_y, width, height, orientation.
std::shared_ptr<Shape> Rectangle::from_yaml(const YAML::Node &node) {
  double cx = node["center_x"].as<double>();
  double cy = node["center_y"].as<double>();
  double width = node["width"].as<double>();
  double height = node["height"].as<double>();
  double orientation = node["orientation"].as<double>();
  return std::make_shared<Rectangle>(Eigen::Vector2d(cx, cy), width, height, orientation);
}
