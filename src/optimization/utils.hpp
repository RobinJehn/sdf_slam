#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "map/map.hpp"
#include "map/utils.hpp"
#include "state/state.hpp"

struct ObjectiveArgs {
  // Scan lines
  double scan_line_factor;  // Factor by which to multiply the scan line
                            // residuals
  int scanline_points;      // Number of points along the scan line
  double step_size;         // Step size between points
  bool both_directions;     // Whether to add points in both directions

  // Scan points
  double scan_point_factor;  // Factor by which to multiply the scan point
                             // residuals

  // Map smoothness
  double smoothness_factor;                   // Factor by which to multiply the smoothness
                                              // term in the objective function
  DerivativeType smoothness_derivative_type;  // Type of derivative approximation for smoothness
                                              // term
  bool project_derivative;                    // Whether to project the derivative onto the
                                              // normal of the point
  int normal_knn;               // Number of nearest neighbours to use for normal estimation
  double normal_search_radius;  // Radius to search for nearest neighbours. If set to 0, the
                                // normal_knn parameter is used instead.

  // Odometry
  double odometry_factor;  // Factor by which to multiply the odometry residuals
};

struct OptimizationArgs {
  int max_iters;          // Maximum number of iterations
  double initial_lambda;  // Initial lambda in Levenberg-Marquardt
  double lambda_factor;   // Factor by which lambda is multiplied or divided
                          // each iteration
  double tolerance;       // Tolerance for stopping criteria
};

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
Eigen::Matrix<double, 3, 6> compute_transformation_derivative_3d(const Eigen::Vector3d &p,
                                                                 const double theta,
                                                                 const double phi,
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
Eigen::Matrix<double, 2, 3> compute_transformation_derivative_2d(const Eigen::Vector2d &p,
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
Eigen::Matrix3d dR_dtheta(const double theta, const double phi, const double psi);

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
template <int Dim>
Eigen::VectorXd flatten(const State<Dim> &state);

/** * @brief Unflattens the given vector into a state object.
 *
 * This function takes a flattened Eigen::VectorXd and converts it back into a
 * State object. The input vector is assumed to be composed of two parts:
 * - The first part represents the map.
 * - The second part represents the transformation.
 *
 * @tparam Dim The dimension of the state.
 * @param flattened_state The flattened Eigen::VectorXd to be unflattened.
 * @param initial_frame The initial frame of the map.
 * @param map_args The arguments used to create the map.
 * @return An unflattened State object.
 */
template <int Dim>
State<Dim> unflatten(const Eigen::VectorXd &flattened_state,
                     const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame,
                     const MapArgs<Dim> &map_args);

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
std::array<typename Map<Dim>::index_t, (1 << Dim)> get_interpolation_point_indices(
    const Eigen::Matrix<double, Dim, 1> &p, const Map<Dim> &map);

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
Eigen::Matrix<double, (1 << Dim), 1> get_interpolation_weights(
    const Eigen::Matrix<double, Dim, 1> &p, const Map<Dim> &map);

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
std::pair<std::array<typename Map<Dim>::index_t, (1 << Dim)>, Eigen::Matrix<double, (1 << Dim), 1>>
get_interpolation_values(const Eigen::Matrix<double, Dim, 1> &p, const Map<Dim> &map);

/**
 * @brief Generates a set of points in global frame and their corresponding
 * desired values.
 *
 * This function generates a specified number of points and their desired values
 * based on the given state and point clouds. The points are generated in either
 * one or both directions with a specified step size.
 *
 * @param state The current state from which points are generated.
 * @param point_clouds A vector of point clouds in scanner frame used to
 * generate the points.
 * @param number_of_points The number of points to generate.
 * @param both_directions A boolean flag indicating whether to generate points
 *                        in both directions.
 * @param step_size The step size used for generating points.
 *
 * @return A vector of pairs, where each pair consists of a global point and its corresponding
 * desired value (double).
 */
template <int Dim>
std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>> generate_points_and_desired_values(
    const State<Dim> &state,
    const std::vector<
        pcl::PointCloud<typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const ObjectiveArgs &objective_args);

/**
 * @brief Converts a 2D point cloud to a 3D point cloud.
 *
 * This function takes a point cloud with 2D points (pcl::PointXY) and converts it to a point cloud
 * with 3D points (pcl::PointXYZ). The z-coordinate of the resulting 3D points will be set to zero.
 *
 * @param cloud_2d A pointer to the input point cloud containing 2D points.
 * @return A pointer to the output point cloud containing 3D points.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d_to_3d(pcl::PointCloud<pcl::PointXY>::Ptr cloud_2d);

/**
 * @brief Compute the analytical derivative of a map at a given point with
 * respect to the point.
 *
 * @tparam Dim
 * @param map
 * @param point Point at which to get the derivative
 * @return dDF_dPoint
 */
template <int Dim>
Eigen::Matrix<double, Dim, 1> compute_analytical_derivative(
    const Map<Dim> &map, const Eigen::Matrix<double, Dim, 1> &point);

/**
 * @brief Approximate the derivative with respect to the point by interpolating
 * the derivative values at the nearest grid points.
 *
 * @tparam Dim
 * @param derivatives Derivative with respect to each dimension
 * @param point Point at which to get the derivative
 * @return dDF_dPoint
 */
template <int Dim>
Eigen::Matrix<double, Dim, 1> compute_approximate_derivative(
    const std::array<Map<Dim>, Dim> &derivatives, const Eigen::Matrix<double, Dim, 1> &point);

/**
 * @brief Computes the derivative of a transformation matrix with respect to a
 * given point.
 *
 * @param point The point in space for which the transformation derivative is
 * computed. Has to be in global frame.
 * @param transform The affine transformation applied to the scanner point.
 * @param numerical If true, the derivative is computed numerically.
 */
template <int Dim>
Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)> compute_transformation_derivative(
    const Eigen::Matrix<double, Dim, 1> &point,
    const Eigen::Transform<double, Dim, Eigen::Affine> &transform, const bool numerical = false);

/**
 * @brief Computes how the residual changes with respect to the transformation.
 *
 * @param map_derivatives The derivatives of the map.
 * @param point The point at which to compute the derivative.
 * @param transform The transformation to compute the derivative for.
 * @param numerical Whether to compute the point transform derivative
 * numerically.
 */
template <int Dim>
Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)> compute_dResidual_dTransform(
    const std::array<Map<Dim>, Dim> &map_derivatives, const Eigen::Matrix<double, Dim, 1> &point,
    const Eigen::Transform<double, Dim, Eigen::Affine> &transform, const bool numerical = false);

template <int Dim>
Eigen::Matrix<double, Dim, 1> compute_dGrad_dNeighbour(const typename Map<Dim>::index_t &index,
                                                       const typename Map<Dim>::index_t &neighbour,
                                                       const std::array<double, Dim> &grid_size,
                                                       const std::array<int, Dim> &num_points);

/**
 * @brief Computes the derivative of roughness with respect to the map.
 *
 * This function calculates the gradient of the roughness metric with respect to the map
 * derivatives provided. The roughness metric is a measure of the variability or
 * irregularity in the map data.
 *
 * @tparam Dim The dimension of the map (e.g., 2 for 2D maps, 3 for 3D maps).
 * @param map_derivatives An array containing the derivatives of the map. Each element
 *                        in the array is a Map object representing the derivative of
 *                        the map with respect to one of the dimensions.
 * @return A vector of doubles representing the computed derivatives of the roughness
 *         with respect to the map.
 */
template <int Dim>
std::vector<double> compute_dRoughness_dMap(const std::array<Map<Dim>, Dim> &map_derivatives);

/**
 * @brief Computes 2D normals for a given point cloud.
 *
 * This function calculates the normals for a 2D point cloud using a specified radius for the
 * search. Assumes that the cloud is scanned from position (0, 0). All normals will be pointing
 * towards the origin.
 *
 * @param cloud A pointer to the input point cloud
 * @param search_radius If different from zero, the normal estimation will use the radius to find
 * out which points it take for line estimation.
 *
 * @param k If different from zero, the normal estimation will find the kth closest points for
 * computing line estimation.
 *
 * @return A pointer to the computed normals
 */
pcl::PointCloud<pcl::Normal>::Ptr compute_normals_2d(
    const pcl::PointCloud<pcl::PointXY>::Ptr &cloud, double search_radius = 0, int k = 3);

/**
 * @brief Computes the normals of a 3D point cloud.
 *
 * This function takes a 3D point cloud and computes the normals for each point
 * in the cloud using a specified radius for the neighborhood search.
 *
 * @param cloud A pointer to the input point cloud
 * @param search_radius The radius used for the neighborhood search to compute the normals.
 * @return A pointer to the point cloud containing the computed normals
 * (pcl::PointCloud<pcl::Normal>).
 */
pcl::PointCloud<pcl::Normal>::Ptr compute_normals_3d(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double search_radius = 0.2);

template <int Dim>
using PointType = typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type;

/**
 * @brief Computes the derivative of a 3D transformation.
 * Assumes R = Rz(psi) * Ry(phi) * Rx(theta)
 *
 * @param p Point at which to compute the derivative. Has to be in scanner
 * frame.
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
Eigen::Matrix<double, 3, 6> compute_transformation_derivative_3d_numerical(const Eigen::Vector3d &p,
                                                                           const double theta,
                                                                           const double phi,
                                                                           const double psi);

/**
 * @brief Computes the derivative of a 2D transformation.
 *
 * @param p Point at which to compute the derivative. Has to be in scanner
 * frame.
 * @param theta Rotation angle
 * @return Eigen::Matrix<double, 2, 3> A matrix representing the derivative of
 *         the transformation. The columns of the matrix represent:
 *         - Column 0: Derivative with respect to x translation
 *         - Column 1: Derivative with respect to y translation
 *         - Column 2: Derivative with respect to rotation (theta)
 */
Eigen::Matrix<double, 2, 3> compute_transformation_derivative_2d_numerical(const Eigen::Vector2d &p,
                                                                           const double theta);

template <int Dim>
Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)>
compute_derivative_map_value_wrt_transformation_numerical(
    const State<Dim> &state, const Eigen::Matrix<double, Dim, 1> &point,
    const Eigen::Transform<double, Dim, Eigen::Affine> &transform, const double epsilon = 1e-8);

/**
 * @brief Template for utility functions or classes that operate on a specified dimension.
 *
 * @tparam Dim The dimension of the space in which the utility functions or classes operate.
 */
template <int Dim>
typename pcl::PointCloud<PointType<Dim>>::Ptr combine_scans(
    const std::vector<typename pcl::PointCloud<PointType<Dim>>::Ptr> &scans);

/**
 * @brief Transforms a set of local point clouds to the global coordinate system.
 *
 * This function takes a vector of transformations and a vector of point clouds,
 * and applies each transformation to the corresponding point cloud to convert
 * it from the local coordinate system to the global coordinate system.
 *
 * @tparam Dim The dimension of the point clouds and transformations (e.g., 2 for 2D, 3 for 3D).
 *
 * @param transformations A vector of Eigen::Transform objects representing the transformations
 *                        from local to global coordinates. The size of this vector should match
 *                        the size of the scans vector.
 * @param scans A vector of pointers to pcl::PointCloud objects representing the local point clouds.
 *              Each point cloud will be transformed to the global coordinate system using the
 *              corresponding transformation from the transformations vector.
 *
 * @return A vector of pointers to pcl::PointCloud objects representing the point clouds in the
 *         global coordinate system.
 */
template <int Dim>
std::vector<typename pcl::PointCloud<PointType<Dim>>::Ptr> local_to_global(
    const std::vector<Eigen::Transform<double, Dim, Eigen::Affine>> transformations,
    const std::vector<typename pcl::PointCloud<PointType<Dim>>::Ptr> &scans);

/**
 * @brief Transforms a set of 2D normal point clouds from local to global coordinates.
 *
 * This function takes a vector of 2D affine transformations and a corresponding vector of
 * point clouds containing normals, and applies each transformation to the respective point
 * cloud to convert the normals from local to global coordinates. Since normals are 3d vectors,
 * the z-component is assumed to be zero.
 *
 * We assume that the normal cloud has the same order as the cloud returned by
 * scans_to_global_pcl_2d.
 *
 * @param transformations 2D affine transformations to be applied to each point cloud.
 * @param scans A vector of point clouds containing normals to be transformed.
 * Each point cloud in this vector corresponds to a transformation in the `transformations` vector.
 *
 * @return A pointer to a new point cloud containing the transformed normals in global coordinates.
 */
pcl::PointCloud<pcl::Normal>::Ptr local_to_global_normals_2d(
    const std::vector<Eigen::Transform<double, 2, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &scans);

/**
 * @brief Computes the normals for a set of 2D point clouds in a global coordinate frame.
 *
 * This function takes a vector of 2D point clouds and their corresponding transformations,
 * and computes the normals for each point cloud in the global coordinate frame.
 *
 * @param scans A vector of shared pointers to 2D point clouds (pcl::PointCloud<pcl::PointXY>::Ptr).
 * @param transformations A vector of 2D affine transformations (Eigen::Transform<double, 2,
 * Eigen::Affine>) that represent the pose of each point cloud in the global coordinate frame.
 * @param search_radius The radius used for the neighborhood search to compute the normals. If set
 * to zero, the k parameter is used instead.
 * @param k The number of nearest neighbors to use for normal estimation.
 *
 * @return A shared pointer to a point cloud of normals (pcl::PointCloud<pcl::Normal>::Ptr) computed
 *         for the input point clouds in the global coordinate frame.
 */
pcl::PointCloud<pcl::Normal>::Ptr compute_normals_global_2d(
    const std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> &scans,
    const std::vector<Eigen::Transform<double, 2, Eigen::Affine>> &transformations,
    double search_radius, int k);

/**
 * @brief Converts a vector of pcl::PointCloud<pcl::PointXY> to a vector of
 * pcl::PointCloud<pcl::PointXY>::Ptr.
 *
 * @param point_clouds A vector of pcl::PointCloud<pcl::PointXY> objects to be converted.
 * @return std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> A vector of shared pointers to
 * pcl::PointCloud<pcl::PointXY> objects.
 */
std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> cloud_to_cloud_ptr(
    const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds);
