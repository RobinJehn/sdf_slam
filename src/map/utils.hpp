#pragma once
#include <Eigen/Dense>

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
float trilinear_interpolation(const Eigen::Vector3f &p,
                              const Eigen::Vector3f &p_floor, const float dx,
                              const float dy, const float dz, const float c000,
                              const float c100, const float c010,
                              const float c110, const float c001,
                              const float c101, const float c011,
                              const float c111);