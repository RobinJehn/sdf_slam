#include "utils.hpp"

double trilinear_interpolation(const Eigen::Vector3d &p,
                               const Eigen::Vector3d &p_floor, const double dx,
                               const double dy, const double dz,
                               const double c000, const double c100,
                               const double c010, const double c110,
                               const double c001, const double c101,
                               const double c011, const double c111) {
  const double xd = (p.x() - p_floor.x()) / dx;
  const double yd = (p.y() - p_floor.y()) / dy;
  const double zd = (p.z() - p_floor.z()) / dz;

  const double c00 = c000 * (1 - xd) + c100 * xd;
  const double c10 = c010 * (1 - xd) + c110 * xd;
  const double c01 = c001 * (1 - xd) + c101 * xd;
  const double c11 = c011 * (1 - xd) + c111 * xd;

  const double c0 = c00 * (1 - yd) + c10 * yd;
  const double c1 = c01 * (1 - yd) + c11 * yd;

  return c0 * (1 - zd) + c1 * zd;
}

double bilinear_interpolation(const Eigen::Vector2d &p,
                              const Eigen::Vector2d &p_floor, const double dx,
                              const double dy, const double c00,
                              const double c10, const double c01,
                              const double c11) {
  const double xd = (p.x() - p_floor.x()) / dx;
  const double yd = (p.y() - p_floor.y()) / dy;

  const double c0 = c00 * (1 - xd) + c10 * xd;
  const double c1 = c01 * (1 - xd) + c11 * xd;

  return c0 * (1 - yd) + c1 * yd;
}