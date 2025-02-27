#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <filesystem>
#include <iostream>

#include "config/utils.hpp"
#include "scan/scan.hpp"

namespace sfs = std::filesystem;

int main(int argc, char** argv) {
  GenerateScanArgs args = setup_generate_scan_args(sfs::path("../config/generate_scans.yml"));

  std::vector<Eigen::Vector2d> scanner_positions;
  std::vector<double> thetas;
  if (args.use_scan_locations) {
    scanner_positions = args.scanner_positions;
    thetas = args.thetas;
  } else {
    for (int i = 0; i < args.number_of_scans; ++i) {
      scanner_positions.push_back(args.initial_position + i * args.delta_position);
      thetas.push_back(args.initial_theta + i * args.delta_theta);
    }
  }

  // Generate scans
  YAML::Node config = YAML::LoadFile("../config/generate_scans.yml");
  YAML::Node scene_config = config["scene"];
  Scene scene = Scene::from_yaml(scene_config);
  const std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> scans = create_scans(
      scene, scanner_positions, thetas, args.num_points, args.angle_range, args.max_range);

  const sfs::path base_dir = sfs::path("../data/") / args.output_directory;
  if (sfs::exists(base_dir)) {
    sfs::remove_all(base_dir);
  }
  sfs::create_directories(base_dir);

  const sfs::path scanner_info_file_path = base_dir / "scanner_info.txt";
  std::ofstream scanner_info_file(scanner_info_file_path.string(), std::ios::app);
  for (size_t i = 0; i < scans.size(); ++i) {
    std::ostringstream oss;
    oss << std::setw(std::to_string(scans.size() - 1).length()) << std::setfill('0') << i;
    const sfs::path scan_file = base_dir / ("scan" + oss.str() + ".pcd");
    pcl::io::savePCDFileASCII(scan_file.string(), *(scans[i]));

    if (scanner_info_file.is_open()) {
      scanner_info_file << scanner_positions[i].x() << " " << scanner_positions[i].y() << " "
                        << thetas[i] << std::endl;
    } else {
      std::cerr << "Unable to open file to save scanner information." << std::endl;
    }
  }
  scanner_info_file.close();

  const sfs::path scene_info_file_path = base_dir / "scene_info.txt";
  std::ofstream scene_info_file(scene_info_file_path.string());
  if (scene_info_file.is_open()) {
    scene_info_file << scene.to_string();
  } else {
    std::cerr << "Unable to open file to save scene information." << std::endl;
  }

  std::cout << "Scans have been generated and saved to disk." << std::endl;
  return 0;
}
