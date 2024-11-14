#include "utils.hpp"
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace sfs = std::filesystem;

Scans read_scans(const std::filesystem::path &dir) {
  Scans scans;

  std::vector<sfs::directory_entry> entries;
  std::copy(sfs::directory_iterator(dir), sfs::directory_iterator(),
            std::back_inserter(entries));

  // Ensures that scan0 comes before scan1, etc.
  std::sort(entries.begin(), entries.end(),
            [](const sfs::directory_entry &a, const sfs::directory_entry &b) {
              return a.path().filename().string() <
                     b.path().filename().string();
            });

  for (const auto &entry : entries) {
    if (entry.path().extension() == ".pcd") {
      pcl::PointCloud<pcl::PointXY> cloud;
      if (pcl::io::loadPCDFile(entry.path().string(), cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", entry.path().string().c_str());
        continue;
      }
      scans.scans.push_back(cloud);
    } else if (entry.path().filename() == "scanner_info.txt") {
      std::ifstream inFile(entry.path().string());
      if (inFile.is_open()) {
        float x, y, theta;
        while (inFile >> x >> y >> theta) {
          const Eigen::Vector2d scanner_position(x, y);
          const Eigen::Transform<double, 2, Eigen::Affine> frame =
              Eigen::Translation<double, 2>(scanner_position) *
              Eigen::Rotation2D<double>(theta);
          scans.frames.push_back(frame);
        }
        inFile.close();
      } else {
        std::cerr << "Unable to open file to read scanner information."
                  << std::endl;
      }
    }
  }

  return scans;
}
