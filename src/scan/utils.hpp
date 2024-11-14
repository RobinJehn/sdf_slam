#pragma once
#include <filesystem>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

struct Scans {
  std::vector<pcl::PointCloud<pcl::PointXY>> scans;
  std::vector<Eigen::Transform<double, 2, Eigen::Affine>> frames;
};

/**
 * @brief Provide a directory with scans scan[number].pcd and scanner_info.txt.
 *
 * @param dir
 * @return Scans
 */
Scans read_scans(const std::filesystem::path &dir);