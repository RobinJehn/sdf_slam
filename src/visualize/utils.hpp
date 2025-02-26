#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

#include "config/utils.hpp"
#include "map/utils.hpp"

template <int Dim>
void plotPointsWithValuesPCL(
    const std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>> &points_with_values);

void visualize_map(const Eigen::VectorXd &params,
                   const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                   const MapArgs<2> &map_args,
                   const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
                   const VisualizationArgs &vis_args);

/**
 * @brief Display the map with the points drawn on it
 *
 * @param map The map to display
 * @param points The points to draw on the map
 * @param min_coords The minimum coordinates of the map
 * @param max_coords The maximum coordinates of the map
 * @param vis_args Visualization arguments
 */
void display_map(const Eigen::MatrixXd &map, const std::vector<Eigen::Vector2d> &points,
                 const Eigen::Vector2d &min_coords, const Eigen::Vector2d &max_coords,
                 const VisualizationArgs &vis_args);

/**
 * @brief Draw the points on the image
 *
 * @param image The image to draw on
 * @param points The points to draw in global frame
 * @param min_coords Minimum coordinates of the map
 * @param max_coords Maximum coordinates of the map
 * @param scale Factor to scale the points from global to image frame
 */
void overlay_points(cv::Mat &image, const std::vector<Eigen::Vector2d> &points,
                    const Eigen::Vector2d &min_coords, const Eigen::Vector2d &max_coords,
                    const Eigen::Vector2d &scale);

/**
 * @brief Turning a map into an image. Applies a color map
 *
 * @param map The map values
 * @param output_width Image width
 * @param output_height Image height
 * @return cv::Mat
 */
cv::Mat mapToImage(const Eigen::MatrixXd &map, int output_width, int output_height);

/**
 * @brief Apply the transformations to the scans
 *
 * @tparam Dim
 * @param transformations The transformations for each scan to global frame
 * @param scans The scans to transform
 *
 * @return List of points in global frame
 */
template <int Dim>
std::vector<Eigen::Matrix<double, Dim, 1>> scans_to_global_eigen(
    const std::vector<Eigen::Transform<double, Dim, Eigen::Affine>> &transformations,
    const std::vector<
        pcl::PointCloud<typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &scans);

/**
 * @brief Converts a series of 2D point cloud scans to a global point cloud using given
 * transformations.
 *
 * This function takes a vector of transformations and a vector of 2D point cloud scans, and applies
 * each transformation to the corresponding scan to produce a global point cloud in 2D space.
 *
 * @param transformations A vector of Eigen::Transform objects representing the transformations to
 * be applied to each scan. Each transformation should be an affine transformation in 2D space.
 * @param scans A vector of pcl::PointCloud<pcl::PointXY> objects representing the 2D point cloud
 * scans to be transformed and combined into a global point cloud.
 * @return A pointer to the resulting global point cloud in 2D space.
 */
pcl::PointCloud<pcl::PointXY>::Ptr scans_to_global_pcl_2d(
    const std::vector<Eigen::Transform<double, 2, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans);
