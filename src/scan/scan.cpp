#include "scan.hpp"

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

pcl::PointCloud<pcl::PointXY>::Ptr create_scan(const Scene &scene,
                                               const Eigen::Vector2d &scanner_position,
                                               const double theta_scanner,  //
                                               const double angle_range,
                                               const int num_points,  //
                                               const double max_range) {
  pcl::PointCloud<pcl::PointXY>::Ptr scan = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
  scan->points.reserve(num_points);

  for (int i = 0; i < num_points; ++i) {
    double angle = theta_scanner - angle_range / 2 + i * (angle_range / (num_points - 1));
    Eigen::Vector2d intersection;
    bool hit = scene.intersect_ray(scanner_position, angle, intersection);
    if (!hit) {
      // No hit: set the intersection at maximum range.
      intersection =
          scanner_position + max_range * Eigen::Vector2d(std::cos(angle), std::sin(angle));
    }
    pcl::PointXY pcl_point;
    pcl_point.x = intersection.x();
    pcl_point.y = intersection.y();
    scan->points.push_back(pcl_point);
  }
  scan->width = static_cast<uint32_t>(scan->points.size());
  scan->height = 1;

  // Transform the scan into the scanner’s coordinate frame.
  Eigen::Translation<double, 2> translation(scanner_position.x(), scanner_position.y());
  Eigen::Rotation2Dd rotation(theta_scanner);
  Eigen::Transform<double, 2, Eigen::Affine> transform = translation * rotation;
  pcl::transformPointCloud(*scan, *scan, transform.inverse().template cast<float>());

  return scan;
}

std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> create_scans(
    const Scene &scene,  //
    const std::vector<Eigen::Vector2d> &scanner_positions,
    const std::vector<double> &thetas,  //
    const int num_points,               //
    const double angle_range,           //
    const double max_range) {
  std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> scans;
  for (int i = 0; i < scanner_positions.size(); ++i) {
    scans.push_back(
        create_scan(scene, scanner_positions[i], thetas[i], angle_range, num_points, max_range));
  }

  return scans;
}
