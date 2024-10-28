#include "state/state.hpp"
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

/**
 * @brief Flattens the given state into a single vector.
 *
 * This function takes a state object and converts it into a single
 * Eigen::VectorXd. The resulting vector is composed of two parts:
 * - The first part represents the map.
 * - The second part represents the transformation.
 *
 * @tparam Dim The dimension of the state.
 * @param state The state object to be flattened.
 * @return A flattened Eigen::VectorXd representing the state.
 */
template <int Dim> Eigen::VectorXd flatten(const State<Dim> &state);

/** * @brief Unflattens the given vector into a state object.
 *
 * This function takes a flattened Eigen::VectorXd and converts it back into a
 * State object. The input vector is assumed to be composed of two parts:
 * - The first part represents the map.
 * - The second part represents the transformation.
 *
 * @tparam Dim The dimension of the state.
 * @param flattened_state The flattened Eigen::VectorXd to be unflattened.
 * @param num_points The number of points in each dimension.
 * @param min_coords The minimum coordinates in each dimension.
 * @param max_coords The maximum coordinates in each dimension.
 * @return An unflattened State object.
 */
template <int Dim>
State<Dim> unflatten(const Eigen::VectorXd &flattened_state,
                     const std::array<int, Dim> &num_points,
                     const Eigen::Matrix<double, Dim, 1> &min_coords,
                     const Eigen::Matrix<double, Dim, 1> &max_coords);

template <int Dim>
int map_index_to_flattened_index(const std::array<int, Dim> &num_points,
                                 const typename Map<Dim>::index_t &index);

/**
 * @brief This function returns the index of the grid cells that are used to
 * interpolate the value at the given point `p`. The number of interpolation
 * points is equal to 2^Dim.
 *
 * @tparam Dim The dimension of the space.
 * @param p The point for which interpolation points are to be computed.
 * @param map The map in which the point `p` is located.
 * @return An array of interpolation points.
 */
template <int Dim>
std::array<typename Map<Dim>::index_t, (1 << Dim)>
get_interpolation_point_indices(const Eigen::Matrix<double, Dim, 1> &p,
                                const Map<Dim> &map);

/**
 * @brief Computes the interpolation weights for a given point in a map.
 *
 * This function calculates the interpolation weights for a point `p` in a
 * map of dimension `Dim`. The weights are used for interpolating values
 * within the map.
 *
 * @tparam Dim The dimension of the map and the point.
 * @param p The point for which to compute the interpolation weights.
 * @param map The map in which the point is located.
 * @return An Eigen::Matrix of size `Dim` containing the interpolation weights.
 */
template <int Dim>
Eigen::Matrix<double, (1 << Dim), 1>
get_interpolation_weights(const Eigen::Matrix<double, Dim, 1> &p,
                          const Map<Dim> &map);

/** @brief Computes the interpolation values for a given point in a map.
 *
 * This function calculates the interpolation indices and corresponding weights
 * for a point `p` in a map of dimension `Dim`. The interpolation is performed
 * using a linear method.
 *
 * @tparam Dim The dimension of the map.
 * @param p The point for which interpolation values are to be computed.
 * @param map The map in which the point `p` is located.
 * @return A pair consisting of:
 *         - An array of indices representing the corners of the interpolation
 * cube.
 *         - A vector of weights corresponding to each corner.
 */
template <int Dim>
std::pair<std::array<typename Map<Dim>::index_t, (1 << Dim)>,
          Eigen::Matrix<double, (1 << Dim), 1>>
get_interpolation_values(const Eigen::Matrix<double, Dim, 1> &p,
                         const Map<Dim> &map);

/**
 * @brief Generates a set of points and their corresponding desired values.
 *
 * This function generates a specified number of points and their desired values
 * based on the given state and point clouds. The points are generated in either
 * one or both directions with a specified step size.
 *
 * @tparam Dim The dimensionality of the state and points (2 or 3).
 * @param state The current state from which points are generated.
 * @param point_clouds A vector of point clouds in scanner frame used to
 * generate the points.
 * @param number_of_points The number of points to generate.
 * @param both_directions A boolean flag indicating whether to generate points
 *                        in both directions.
 * @param step_size The step size used for generating points.
 * @return A vector of pairs, where each pair consists of a point
 * (Eigen::Matrix) and its corresponding desired value (double).
 */
template <int Dim>
std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
generate_points_and_desired_values(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size);

template <int Dim>
Eigen::VectorXd compute_residuals(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size);

template <int Dim>
Eigen::VectorXd
objective_vec(const State<Dim> &state,
              const std::vector<pcl::PointCloud<typename std::conditional<
                  Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>> &point_clouds,
              const int number_of_points, const bool both_directions,
              const double step_size);