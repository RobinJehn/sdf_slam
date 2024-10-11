#include <Eigen/Dense>

/**
 * @brief Computes the derivative of a 3D transformation.
 * Assumes R = Rz(psi) * Ry(phi) * Rx(theta)
 *
 * @param p Point at which to compute the derivative.
 * @param theta Rotation angle around the x-axis.
 * @param phi Rotation angle around the y-axis.
 * @param psi Rotation angle around the z-axis.
 * @return Eigen::Matrix<double, 3, 6> A matrix representing the derivative of
 *         the transformation. The columns of the matrix represent:
 *         - Column 0: Derivative with respect to x translation.
 *         - Column 1: Derivative with respect to y translation.
 *         - Column 2: Derivative with respect to z translation.
 *         - Column 3: Derivative with respect to rotation around the x-axis
 * (theta).
 *         - Column 4: Derivative with respect to rotation around the y-axis
 * (phi).
 *         - Column 5: Derivative with respect to rotation around the z-axis
 * (psi).
 */
Eigen::Matrix<double, 3, 6>
compute_transformation_derivative_3d(const Eigen::Vector3d &p,
                                     const double theta, const double phi,
                                     const double psi);

/**
 * @brief Computes the derivative of a 2D transformation.
 *
 * @param p Point at which to compute the derivative
 * @param theta Rotation angle
 * @return Eigen::Matrix<double, 2, 3> A matrix representing the derivative of
 *         the transformation. The columns of the matrix represent:
 *         - Column 0: Derivative with respect to x translation
 *         - Column 1: Derivative with respect to y translation
 *         - Column 2: Derivative with respect to rotation (theta)
 */
Eigen::Matrix<double, 2, 3>
compute_transformation_derivative_2d(const Eigen::Vector2d &p,
                                     const double theta);

/**
 * @brief Compute the derivative of a 3D rotation matrix w.r.t. yaw (theta).
 * Assumes the rotation matrix is defined as R = Rz(psi) * Ry(phi) * Rx(theta)
 *
 * @param theta
 * @param phi
 * @param psi
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d dR_dtheta(const double theta, const double phi,
                          const double psi);

/**
 * @brief Compute the derivative of a 3D rotation matrix w.r.t. pitch (phi).
 * Assumes the rotation matrix is defined as R = Rz(psi) * Ry(phi) * Rx(theta)
 *
 * @param theta
 * @param phi
 * @param psi
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d dR_dphi(const double theta, const double phi, const double psi);

/**
 * @brief Compute the derivative of a 3D rotation matrix w.r.t. roll (psi).
 * Assumes the rotation matrix is defined as R = Rz(psi) * Ry(phi) * Rx(theta)
 *
 * @param theta
 * @param phi
 * @param psi
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d dR_dpsi(const double theta, const double phi, const double psi);
