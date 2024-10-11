#include "utils.hpp"

Eigen::Matrix<double, 3, 6>
compute_transformation_derivative_3d(const Eigen::Vector3d &p,
                                     const double theta, const double phi,
                                     const double psi) {
  // Derivative matrix (rotation and translation)
  Eigen::Matrix<double, 3, 6> derivative;

  // Translation derivatives
  derivative.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // Rotation derivatives
  derivative.col(3) = dR_dtheta(theta, phi, psi) * p;
  derivative.col(4) = dR_dphi(theta, phi, psi) * p;
  derivative.col(5) = dR_dpsi(theta, phi, psi) * p;

  return derivative;
}

Eigen::Matrix<double, 2, 3>
compute_transformation_derivative_2d(const Eigen::Vector2d &p,
                                     const double theta) {
  Eigen::Matrix<double, 2, 3> derivative;

  // Translation derivatives
  derivative.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();

  // Rotation derivative w.r.t. theta
  Eigen::Matrix2d dR_dtheta;
  dR_dtheta << -sin(theta), -cos(theta), cos(theta), -sin(theta);
  derivative.col(2) = dR_dtheta * p;

  return derivative;
}

Eigen::Matrix3d dR_dtheta(const double theta, const double phi,
                          const double psi) {
  Eigen::Matrix3d dRz_dtheta;

  dRz_dtheta << -sin(theta), -cos(theta), 0, cos(theta), -sin(theta), 0, 0, 0,
      0;

  Eigen::Matrix3d R_y =
      Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Matrix3d R_x =
      Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitX()).toRotationMatrix();

  return dRz_dtheta * R_y * R_x;
}

Eigen::Matrix3d dR_dphi(const double theta, const double phi,
                        const double psi) {
  Eigen::Matrix3d dRy_dphi;

  dRy_dphi << -sin(phi), 0, cos(phi), 0, 0, 0, -cos(phi), 0, -sin(phi);

  Eigen::Matrix3d R_z =
      Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Matrix3d R_x =
      Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitX()).toRotationMatrix();

  return R_z * dRy_dphi * R_x;
}

Eigen::Matrix3d dR_dpsi(const double theta, const double phi,
                        const double psi) {
  Eigen::Matrix3d dRx_dpsi;

  dRx_dpsi << 0, 0, 0, 0, -sin(psi), -cos(psi), 0, cos(psi), -sin(psi);

  Eigen::Matrix3d R_z =
      Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Matrix3d R_y =
      Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()).toRotationMatrix();

  return R_z * R_y * dRx_dpsi;
}