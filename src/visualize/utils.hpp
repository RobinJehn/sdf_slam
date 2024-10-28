#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <utility>
#include <vector>

template <int Dim>
void plotPointsWithValuesPCL(
    const std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
        &points_with_values);

void visualizeMap(const Eigen::VectorXd &params,
                  const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                  const std::array<int, 2> &num_map_points,
                  const Eigen::Vector2d &min_coords,
                  const Eigen::Vector2d &max_coords,
                  const int output_width = 1000,
                  const int output_height = 1000);

/**
 * @brief Display the map with the points drawn on it
 *
 * @param map The map to display
 * @param points The points to draw on the map
 * @param min_coords The minimum coordinates of the map
 * @param max_coords The maximum coordinates of the map
 * @param output_width The width of the output image
 * @param output_height The height of the output image
 */
void displayMapWithPoints(const Eigen::MatrixXd &map,
                          const std::vector<Eigen::Vector2d> &points,
                          const Eigen::Vector2d &min_coords,
                          const Eigen::Vector2d &max_coords,
                          const int output_width = 1000,
                          const int output_height = 1000);

/**
 * @brief Draw the points on the image
 *
 * @param image The image to draw on
 * @param points The points to draw in global frame
 * @param min_coords Minimum coordinates of the map
 * @param max_coords Maximum coordinates of the map
 * @param scale Factor to scale the points from global to image frame
 */
void overlayPoints(cv::Mat &image, const std::vector<Eigen::Vector2d> &points,
                   const Eigen::Vector2d &min_coords,
                   const Eigen::Vector2d &max_coords,
                   const Eigen::Vector2d &scale);

/**
 * @brief Turning a map into an image. Applies a color map
 *
 * @param map The map values
 * @param output_width Image width
 * @param output_height Image height
 * @return cv::Mat
 */
cv::Mat mapToImage(const Eigen::MatrixXd &map, int output_width,
                   int output_height);

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
std::vector<Eigen::Matrix<double, Dim, 1>>
scan_to_global(const std::vector<Eigen::Transform<double, Dim, Eigen::Affine>>
                   &transformations,
               const std::vector<pcl::PointCloud<typename std::conditional<
                   Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>> &scans);
