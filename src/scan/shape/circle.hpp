#pragma once
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "shape.hpp"

/**
 * @brief A circle defined by its center and radius.
 */
class Circle : public Shape {
 public:
  Circle(const Eigen::Vector2d &center, double radius);

  /**
   * @brief Find the intersection between the ray and the circle.
   */
  bool intersect_ray(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d &intersection,
                     double &t) const override;

  /**
   * @brief Compute the distance from a point to the circle.
   */
  double distance(const Eigen::Vector2d &p) const override;

  /**
   * @return A string representation of the object.
   *
   * Format: "circle center_x center_y radius"
   */
  std::string to_string() const override;

  /**
   * @brief Creates a Shape object from a string representation.
   *
   * Format: "circle center_x center_y radius"
   *
   * @param str The string representation of the Shape.
   * @return A shared pointer to the created Shape object.
   */
  static std::shared_ptr<Shape> from_string(const std::string &str);

  /**
   * @brief Creates a Shape object from a YAML node.
   *
   * Format: "-type: circle -center_x: -center_y: -radius:"
   *
   * @param node The YAML node.
   * @return A shared pointer to the created Shape object.
   */
  static std::shared_ptr<Shape> from_yaml(const YAML::Node &node);

 private:
  Eigen::Vector2d center_;
  double radius_;
};
