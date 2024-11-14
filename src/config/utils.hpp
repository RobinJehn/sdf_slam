#pragma once
#include "map/utils.hpp"
#include "optimization/utils.hpp"
#include <filesystem>

struct GeneralArgs {
  std::filesystem::path data_path; // Path to the data directory with the scans
  bool from_ground_truth = false;  // Whether to initialize the map from ground
                                   // truth
};

template <int Dim> struct Args {
  MapArgs<Dim> map_args;
  ObjectiveArgs objective_args;
  OptimizationArgs optimization_args;
  GeneralArgs general_args;
};

template <int Dim>
Args<Dim> setup_from_yaml(const std::filesystem::path &config_path);

struct GenerateScanArgs {
  std::filesystem::path output_directory;
  int number_of_scans;
  double initial_theta;
  Eigen::Vector2d initial_position;
  double delta_theta;
  Eigen::Vector2d delta_position;
};

GenerateScanArgs setup_generate_scan_args(const std::filesystem::path &config_path);