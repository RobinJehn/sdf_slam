#pragma once

#include <Eigen/Dense>
#include <cmath>

/**
 * @brief Abstract base class for shapes.
 */
class Shape {
 public:
  virtual ~Shape() {}

  /**
   * @brief Given a ray (origin + t * [cos(angle), sin(angle)] for t>=0),
   * compute an intersection point if it exists.
   *
   * Outputs:
   *  - intersection: the point of intersection (if any)
   *  - t: the distance along the ray (so that intersection = origin + t * dir)
   *
   * @returns true if an intersection was found (with t>0).
   */
  virtual bool intersect_ray(const Eigen::Vector2d &origin, double angle,
                             Eigen::Vector2d &intersection, double &t) const = 0;

  /**
   * @brief Compute the distance from a point to the shape.
   *
   * @param p The point to compute the distance to.
   * @returns The distance from the point to the shape.
   *          Negative if the point is inside the shape.
   *          Positive if the point is outside the shape.
   */
  virtual double distance(const Eigen::Vector2d &p) const = 0;

  /**
   * @return A string representation of the object.
   */
  virtual std::string to_string() const = 0;
};

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
};

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
   * Format: "Circle center_x center_y radius"
   */
  std::string to_string() const override;

  /**
   * @brief Creates a Shape object from a string representation.
   *
   * Format: "Circle center_x center_y radius"
   *
   * @param str The string representation of the Shape.
   * @return A shared pointer to the created Shape object.
   */
  static std::shared_ptr<Shape> from_string(const std::string &str);

 private:
  Eigen::Vector2d center_;
  double radius_;
};
