#include "utils.hpp"

float trilinear_interpolation(const Eigen::Vector3f &p,
                              const Eigen::Vector3f &p_floor, const float dx,
                              const float dy, const float dz, const float c000,
                              const float c100, const float c010,
                              const float c110, const float c001,
                              const float c101, const float c011,
                              const float c111) {
  const float xd = (p.x() - p_floor.x()) / dx;
  const float yd = (p.y() - p_floor.y()) / dy;
  const float zd = (p.z() - p_floor.z()) / dz;

  const float c00 = c000 * (1 - xd) + c100 * xd;
  const float c10 = c010 * (1 - xd) + c110 * xd;
  const float c01 = c001 * (1 - xd) + c101 * xd;
  const float c11 = c011 * (1 - xd) + c111 * xd;

  const float c0 = c00 * (1 - yd) + c10 * yd;
  const float c1 = c01 * (1 - yd) + c11 * yd;

  return c0 * (1 - zd) + c1 * zd;
}

float bilinear_interpolation(const Eigen::Vector2f &p,
               const Eigen::Vector2f &p_floor, const float dx,
               const float dy, const float c00, const float c10,
               const float c01, const float c11) {
  const float xd = (p.x() - p_floor.x()) / dx;
  const float yd = (p.y() - p_floor.y()) / dy;

  const float c0 = c00 * (1 - xd) + c10 * xd;
  const float c1 = c01 * (1 - xd) + c11 * xd;

  return c0 * (1 - yd) + c1 * yd;
}