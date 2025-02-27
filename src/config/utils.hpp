#pragma once
#include <filesystem>

#include "map/utils.hpp"
#include "optimization/utils.hpp"

struct GeneralArgs {
  std::filesystem::path data_path;  // Path to the data directory with the scans
  bool from_ground_truth = false;   // Whether to initialize the map from ground
                                    // truth
  double initial_value = 0.0;       // Initial value for the map
};

struct VisualizationArgs {
  bool show_points = true;   // Whether to show the points on the map
  bool show_normals = true;  // Whether to show the normals on the map
  int output_width = 1000;   // Width of the output image
  int output_height = 1000;  // Height of the output image
};

template <int Dim>
struct Args {
  MapArgs<Dim> map_args;
  ObjectiveArgs objective_args;
  OptimizationArgs optimization_args;
  VisualizationArgs vis_args;
  GeneralArgs general_args;
};

template <int Dim>
Args<Dim> setup_from_yaml(const std::filesystem::path &config_path);

struct GenerateScanArgs {
  std::filesystem::path output_directory;

  // Scan
  int number_of_scans;
  double angle_range;
  int num_points;
  double max_range;

  // Scan Location
  bool use_scan_locations;
  std::vector<Eigen::Vector2d> scanner_positions;
  std::vector<double> thetas;

  // Location Generation
  double initial_theta;
  Eigen::Vector2d initial_position;
  double delta_theta;
  Eigen::Vector2d delta_position;
};

GenerateScanArgs setup_generate_scan_args(const std::filesystem::path &config_path);
