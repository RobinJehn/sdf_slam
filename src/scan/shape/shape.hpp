#pragma once

#include <Eigen/Dense>
#include <string>

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
