#pragma once
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "shape.hpp"

/**
 * @brief A sinusoid defined by f(x) = sin(x).
 */
class Sinusoid : public Shape {
 public:
  /**
   * @brief Find the intersection between the ray and the curve y = sin(x).
   */
  bool intersect_ray(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d &intersection,
                     double &t) const override;

  /**
   * @brief Compute the distance from a point to the sinusoid.
   */
  double distance(const Eigen::Vector2d &p) const override;

  /**
   * @return A string representation of the object.
   *
   * Sinusoid has no parameters so we simply output its type.
   */
  std::string to_string() const override;

  /**
   * @brief Creates a Shape object from a string representation.
   *
   * Ignores additional parameters (if any) and returns a new Sinusoid.
   *
   * @param str The string representation of the Shape.
   * @return A shared pointer to the created Shape object.
   */
  static std::shared_ptr<Shape> from_string(const std::string &str);

  /**
   * @brief Creates a Shape object from a YAML node.
   *
   * Ignores additional parameters (if any) and returns a new Sinusoid.
   *
   * Format: "-type: sinusoid"
   *
   * @param node The YAML node.
   * @return A shared pointer to the created Shape object.
   */
  static std::shared_ptr<Shape> from_yaml(const YAML::Node &node);
};
