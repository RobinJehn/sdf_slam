#pragma once
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "shape.hpp"

/**
 * @brief A rectangle defined by its center, width, height, and orientation.
 */
class Rectangle : public Shape {
 public:
  /**
   * @brief Construct a new Rectangle object.
   *
   * @param center The center of the rectangle.
   * @param width The width (extent along local x-axis).
   * @param height The height (extent along local y-axis).
   * @param orientation The orientation (in radians) of the rectangle relative to the global frame.
   */
  Rectangle(const Eigen::Vector2d &center, double width, double height, double orientation);

  /**
   * @brief Find the intersection between the ray and the rectangle.
   *
   * This method transforms the ray into the rectangle's local coordinate frame (where it is
   * axis-aligned) and uses the slab method to compute the intersection.
   *
   * @param origin The origin of the ray.
   * @param angle The angle of the ray (in radians).
   * @param intersection The computed intersection point (if any).
   * @param t The distance along the ray to the intersection.
   * @return true if an intersection is found (with t > 0), false otherwise.
   */
  bool intersect_ray(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d &intersection,
                     double &t) const override;

  /**
   * @brief Compute the (signed) distance from a point to the rectangle.
   *
   * If the point is outside the rectangle, returns the Euclidean distance to the boundary.
   * If inside, returns the negative distance to the nearest edge.
   *
   * @param p The query point.
   * @return double The signed distance.
   */
  double distance(const Eigen::Vector2d &p) const override;

  /**
   * @brief Get a string representation of the rectangle.
   *
   * Format: "rectangle center_x center_y width height orientation"
   *
   * @return std::string The string representation.
   */
  std::string to_string() const override;

  /**
   * @brief Create a Rectangle from its string representation.
   *
   * Format: "rectangle center_x center_y width height orientation"
   *
   * @param str The string representation.
   * @return std::shared_ptr<Shape> A shared pointer to the created Rectangle.
   */
  static std::shared_ptr<Shape> from_string(const std::string &str);

  /**
   * @brief Create a Rectangle from a YAML node.
   *
   * Expected keys: center_x, center_y, width, height, orientation.
   *
   * @param node The YAML node.
   * @return std::shared_ptr<Shape> A shared pointer to the created Rectangle.
   */
  static std::shared_ptr<Shape> from_yaml(const YAML::Node &node);

 private:
  Eigen::Vector2d center_;
  double width_;
  double height_;
  double orientation_;  // in radians
};
