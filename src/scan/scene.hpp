#pragma once

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <filesystem>
#include <memory>
#include <vector>

#include "shape/shape.hpp"

/**
 * @brief A scene that holds multiple shapes.
 */
class Scene {
 public:
  void add_shape(const std::shared_ptr<Shape> &shape);

  /**
   * @brief Returns the intersection point of the ray with the closest shape.
   */
  bool intersect_ray(const Eigen::Vector2d &origin, double angle,
                     Eigen::Vector2d &intersection) const;

  /**
   * @brief Retrieves a constant reference to a vector of shared pointers to Shape objects.
   *
   * This function returns a constant reference to a vector containing shared pointers
   * to Shape objects. The vector represents the collection of shapes associated with
   * the current scene.
   *
   * @return A constant reference to the vector of shared pointers to Shape objects.
   */
  const std::vector<std::shared_ptr<Shape>> &get_shapes() const;

  /**
   * @return A string representing the Scene.
   *
   * Format: "Shape1\nShape2\n..."
   */
  std::string to_string() const;

  /**
   * @brief Create a Scene from a string.
   *
   * The string should be formatted as "Shape1\nShape2\n...".
   *
   * @param str The string to parse.
   *
   * @return The Scene object.
   */
  static Scene from_string(const std::string &str);

  /**
   * @brief Create a Scene from a file.
   *
   * The file should be formatted as "Shape1\nShape2\n...".
   *
   * @param path Path to the file.
   *
   * @return The Scene object.
   */
  static Scene from_file(const std::filesystem::path &path);

  /**
   * @brief Create a Scene from a yaml node.
   *
   * The file should be formatted as "Shape1\nShape2\n...".
   *
   * @param node The YAML node.
   *
   * @return The Scene object.
   */
  static Scene from_yaml(const YAML::Node &node);

 private:
  std::vector<std::shared_ptr<Shape>> shapes_;
};
