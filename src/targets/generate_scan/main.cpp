#include "scan/generate.hpp"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main(int argc, char **argv) {
  // Define scanner positions and orientations
  const double theta_scanner_1 = 9 * M_PI / 16;
  const Eigen::Vector2d scanner_position_1 = {3.5, -5};

  const double theta_scanner_2 = 8 * M_PI / 16;
  const Eigen::Vector2d scanner_position_2 = {4, -5};

  std::vector<Eigen::Vector2d> scanner_positions = {scanner_position_1,
                                                    scanner_position_2};
  std::vector<double> thetas = {theta_scanner_1, theta_scanner_2};
  // Generate scans
  auto scans = create_scans(scanner_positions, thetas);

  // Save scans to disk
  pcl::io::savePCDFileASCII("scan1.pcd", scans[0]);
  pcl::io::savePCDFileASCII("scan2.pcd", scans[1]);

  std::cout << "Scans have been generated and saved to disk." << std::endl;
  return 0;
}
