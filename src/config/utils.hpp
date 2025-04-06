#pragma once
#include <filesystem>

#include "map/utils.hpp"
#include "optimization/utils.hpp"

struct GeneralArgs {
  std::filesystem::path data_path;  // Path to the data directory with the scans
  bool from_ground_truth;           // Whether to initialize the map from ground
                                    // truth
  double initial_value;             // Initial value for the map
  bool save_results;                // Whether to save the results
  std::string experiment_name;      // Name of the experiment
};

struct VisualizationArgs {
  bool save_file;         // Whether to save the output image
  bool show_points;       // Whether to show the points on the map
  bool show_normals;      // Whether to show the normals on the map
  bool show_path;         // Whether to show the path of the scanner
  int output_width;       // Width of the output image
  int output_height;      // Height of the output image
  bool clamp_colour_map;  // Whether to clamp the colour map or make it relative to min and max in
                          // the specific image
  double min_value;       // Minimum value for the color map
  double max_value;       // Maximum value for the color map
  bool visualize;         // Whether to visualize the map on each iteration
  bool initial_visualization;  // Whether to visualize the initial map
  bool std_out;                // Whether to print to stdout
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
