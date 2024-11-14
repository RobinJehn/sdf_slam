#include "config/utils.hpp"
#include "scan/generate.hpp"
#include <filesystem>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace sfs = std::filesystem;

int main(int argc, char **argv) {
  GenerateScanArgs args =
      setup_generate_scan_args(sfs::path("../config/generate_scans.yml"));

  std::vector<Eigen::Vector2d> scanner_positions;
  std::vector<double> thetas;
  for (int i = 0; i < args.number_of_scans; ++i) {
    scanner_positions.push_back(args.initial_position +
                                i * args.delta_position);
    thetas.push_back(args.initial_theta + i * args.delta_theta);
  }

  // Generate scans
  const auto scans = create_scans(scanner_positions, thetas);
  const sfs::path base_dir = sfs::path("../data/") / args.output_directory;
  if (sfs::exists(base_dir)) {
    sfs::remove_all(base_dir);
  }
  sfs::create_directories(base_dir);
  const sfs::path info_file = base_dir / "scanner_info.txt";
  std::ofstream outFile(info_file.string(), std::ios::app);
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
