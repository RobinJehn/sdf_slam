#include "scan/generate.hpp"
#include <filesystem>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace sfs = std::filesystem;

int main(int argc, char **argv) {
  // Define scanner positions and orientations
  const double theta_scanner_1 = 9 * M_PI / 16;
  const Eigen::Vector2d scanner_position_1 = {3.5, -5};

  const double theta_scanner_2 = 8 * M_PI / 16;
  const Eigen::Vector2d scanner_position_2 = {4, -5};

  const std::vector<Eigen::Vector2d> scanner_positions = {scanner_position_1,
                                                          scanner_position_2};
  const std::vector<double> thetas = {theta_scanner_1, theta_scanner_2};
  // Generate scans
  const auto scans = create_scans(scanner_positions, thetas);

  // Save Info to disk
  const sfs::path base_dir = "../data/scans";
  const sfs::path info_file = base_dir / "scanner_info.txt";
  std::ofstream outFile(info_file.string(), std::ios::app);
  sfs::create_directories(base_dir);
  for (size_t i = 0; i < scans.size(); ++i) {
    const sfs::path scan_file =
        base_dir / ("scan" + std::to_string(i) + ".pcd");
    pcl::io::savePCDFileASCII(scan_file.string(), scans[i]);

    if (outFile.is_open()) {
      outFile << scanner_positions[i].x() << " " << scanner_positions[i].y()
              << " " << thetas[i] << std::endl;
    } else {
      std::cerr << "Unable to open file to save scanner information."
                << std::endl;
    }
  }
  outFile.close();

  std::cout << "Scans have been generated and saved to disk." << std::endl;
  return 0;
}
