#pragma once
#include <Eigen/Dense>

template <int Dim> struct MapArgs {
  using Vector = Eigen::Matrix<double, Dim, 1>;
  std::array<int, Dim> num_points;
  Vector min_coords;
  Vector max_coords;
};

/**
 * @brief Performs trilinear interpolation within a 3D space defined by eight
 * corner points.
 *
 * This function interpolates the value at a given point (x, y, z) within a cube
 * defined by its eight corner points. For more details on trilinear
 * interpolation, refer to
 * https://en.wikipedia.org/wiki/Trilinear_interpolation.
 *
 * @param p The point at which to interpolate, containing the x, y, and z
 * coordinates.
 * @param p_floor The lower bound point of the interpolation cube.
 * @param dx The difference between the x-coordinates of the upper and lower
 * bounds.
 * @param dy The difference between the y-coordinates of the upper and lower
 * bounds.
 * @param dz The difference between the z-coordinates of the upper and lower
 * bounds.
 * @param c000 The value at the (x_floor, y_floor, z_floor) corner.
 * @param c100 The value at the (x_floor + dx, y_floor, z_floor) corner.
 * @param c010 The value at the (x_floor, y_floor + dy, z_floor) corner.
 * @param c110 The value at the (x_floor + dx, y_floor + dy, z_floor) corner.
 * @param c001 The value at the (x_floor, y_floor, z_floor + dz) corner.
 * @param c101 The value at the (x_floor + dx, y_floor, z_floor + dz) corner.
 * @param c011 The value at the (x_floor, y_floor + dy, z_floor + dz) corner.
 * @param c111 The value at the (x_floor + dx, y_floor + dy, z_floor + dz)
 * corner.
 *
 * @return The trilinearly interpolated value at the point (x, y, z).
 */
double trilinear_interpolation(const Eigen::Vector3d &p,
                               const Eigen::Vector3d &p_floor, const double dx,
                               const double dy, const double dz,
                               const double c000, const double c100,
                               const double c010, const double c110,
                               const double c001, const double c101,
                               const double c011, const double c111);

/**
 * @brief Performs bilinear interpolation within a 2D space defined by four
 * corner points.
 *
 * This function interpolates the value at a given point (x, y) within a square
 * defined by its four corner points. For more details on bilinear
 * interpolation, refer to
 * https://en.wikipedia.org/wiki/Bilinear_interpolation.
 *
 * @param p The point at which to interpolate, containing the x and y
 * coordinates.
 * @param p_floor The lower bound point of the interpolation square.
 * @param dx The difference between the x-coordinates of the upper and lower
 * bounds.
 * @param dy The difference between the y-coordinates of the upper and lower
 * bounds.
 * @param c00 The value at the (x_floor, y_floor) corner.
 * @param c10 The value at the (x_floor + dx, y_floor) corner.
 * @param c01 The value at the (x_floor, y_floor + dy) corner.
 * @param c11 The value at the (x_floor + dx, y_floor + dy) corner.
 *
 * @return The bilinearly interpolated value at the point (x, y).
 */
double bilinear_interpolation(const Eigen::Vector2d &p,
                              const Eigen::Vector2d &p_floor, const double dx,
                              const double dy, const double c00,
                              const double c10, const double c01,
                              const double c11);