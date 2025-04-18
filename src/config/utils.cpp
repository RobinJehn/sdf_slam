#include "utils.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>

namespace sfs = std::filesystem;

template <int Dim>
Args<Dim> setup_from_yaml(const sfs::path &config_path) {
  // Load the YAML file
  YAML::Node config = YAML::LoadFile(config_path.string());

  Args<Dim> args;

  // Read objective_args from YAML
  args.objective_args.scanline_points = config["objective_args"]["scanline_points"].as<int>();
  args.objective_args.step_size = config["objective_args"]["step_size"].as<double>();
  args.objective_args.both_directions = config["objective_args"]["both_directions"].as<bool>();
  args.objective_args.scan_line_factor = config["objective_args"]["scan_line_factor"].as<double>();
  args.objective_args.scan_point_factor =
      config["objective_args"]["scan_point_factor"].as<double>();
  args.objective_args.smoothness_factor =
      config["objective_args"]["smoothness_factor"].as<double>();
  args.objective_args.project_derivative =
      config["objective_args"]["project_derivative"].as<bool>();
  args.objective_args.normal_knn = config["objective_args"]["normal_knn"].as<int>();
  args.objective_args.normal_search_radius =
      config["objective_args"]["normal_search_radius"].as<double>();

  args.objective_args.odometry_factor = config["objective_args"]["odometry_factor"].as<double>();

  const std::string smoothness_derivative_type =
      config["objective_args"]["smoothness_derivative_type"].as<std::string>();
  if (smoothness_derivative_type == "UPWIND") {
    args.objective_args.smoothness_derivative_type = DerivativeType::UPWIND;
  } else if (smoothness_derivative_type == "FORWARD") {
    args.objective_args.smoothness_derivative_type = DerivativeType::FORWARD;
  } else if (smoothness_derivative_type == "CENTRAL") {
    args.objective_args.smoothness_derivative_type = DerivativeType::CENTRAL;
  } else {
    throw std::runtime_error("Invalid smoothness_derivative_type.");
  }

  // Read map_args from YAML
  if constexpr (Dim == 2) {
    args.map_args.num_points = {config["map_args"]["num_points"][0].as<int>(),
                                config["map_args"]["num_points"][1].as<int>()};
    args.map_args.min_coords = Eigen::Vector2d(config["map_args"]["min_coords"][0].as<double>(),
                                               config["map_args"]["min_coords"][1].as<double>());
    args.map_args.max_coords = Eigen::Vector2d(config["map_args"]["max_coords"][0].as<double>(),
                                               config["map_args"]["max_coords"][1].as<double>());
  } else if constexpr (Dim == 3) {
    args.map_args.num_points = {config["map_args"]["num_points"][0].as<int>(),
                                config["map_args"]["num_points"][1].as<int>(),
                                config["map_args"]["num_points"][2].as<int>()};
    args.map_args.min_coords = Eigen::Vector3d(config["map_args"]["min_coords"][0].as<double>(),
                                               config["map_args"]["min_coords"][1].as<double>(),
                                               config["map_args"]["min_coords"][2].as<double>());
    args.map_args.max_coords = Eigen::Vector3d(config["map_args"]["max_coords"][0].as<double>(),
                                               config["map_args"]["max_coords"][1].as<double>(),
                                               config["map_args"]["max_coords"][2].as<double>());
  }

  // Read optimization_args from YAML
  args.optimization_args.max_iters = config["optimization_args"]["max_iters"].as<int>();
  args.optimization_args.initial_lambda =
      config["optimization_args"]["initial_lambda"].as<double>();
  args.optimization_args.tolerance = config["optimization_args"]["tolerance"].as<double>();
  args.optimization_args.lambda_factor = config["optimization_args"]["lambda_factor"].as<double>();

  // Read visualization_args from YAML
  args.vis_args.output_width = config["visualization_args"]["output_width"].as<int>();
  args.vis_args.output_height = config["visualization_args"]["output_height"].as<int>();
  args.vis_args.show_normals = config["visualization_args"]["show_normals"].as<bool>();
  args.vis_args.show_points = config["visualization_args"]["show_points"].as<bool>();
  args.vis_args.clamp_colour_map = config["visualization_args"]["clamp_colour_map"].as<bool>();
  args.vis_args.min_value = config["visualization_args"]["min_value"].as<double>();
  args.vis_args.max_value = config["visualization_args"]["max_value"].as<double>();
  args.vis_args.show_path = config["visualization_args"]["show_path"].as<bool>();
  args.vis_args.save_file = config["visualization_args"]["save_file"].as<bool>();
  args.vis_args.visualize = config["visualization_args"]["visualize"].as<bool>();
  args.vis_args.initial_visualization =
      config["visualization_args"]["initial_visualization"].as<bool>();
  args.vis_args.std_out = config["visualization_args"]["std_out"].as<bool>();

  // Read general_args from YAML
  args.general_args.from_ground_truth = config["general_args"]["from_ground_truth"].as<bool>();
  args.general_args.save_results = config["general_args"]["save_results"].as<bool>();
  args.general_args.initial_value = config["general_args"]["initial_value"].as<double>();
  args.general_args.data_path = sfs::path(config["general_args"]["data_path"].as<std::string>());
  args.general_args.experiment_name = config["general_args"]["experiment_name"].as<std::string>();

  if (args.objective_args.both_directions && args.objective_args.scanline_points % 2 != 0) {
    throw std::runtime_error("scanline_points must be even when both_directions is true.");
  }

  if (args.objective_args.scanline_points < 0) {
    throw std::runtime_error("scanline_points must be greater than or equal to 0.");
  }

  if (args.objective_args.step_size <= 0) {
    throw std::runtime_error("step_size must be greater than 0.");
  }

  if (args.objective_args.scan_line_factor < 0) {
    throw std::runtime_error("scan_line_factor must be greater than or equal to 0.");
  }

  if (args.objective_args.scan_point_factor < 0) {
    throw std::runtime_error("scan_point_factor must be greater than or equal to 0.");
  }

  return args;
}

GenerateScanArgs setup_generate_scan_args(const std::filesystem::path &config_path) {
  // Load the YAML file
  YAML::Node config = YAML::LoadFile(config_path.string());

  GenerateScanArgs args;
  args.output_directory = sfs::path(config["output_directory"].as<std::string>());
  args.number_of_scans = config["number_of_scans"].as<int>();
  args.initial_theta = config["initial_theta"].as<double>();
  args.initial_position = Eigen::Vector2d(config["initial_position"][0].as<double>(),
                                          config["initial_position"][1].as<double>());
  args.delta_theta = config["delta_theta"].as<double>();
  args.delta_position = Eigen::Vector2d(config["delta_position"][0].as<double>(),
                                        config["delta_position"][1].as<double>());
  args.angle_range = config["angle_range"].as<double>();
  args.num_points = config["num_points"].as<int>();
  args.max_range = config["max_range"].as<double>();

  args.use_scan_locations = config["use_scan_locations"].as<bool>();
  if (args.use_scan_locations) {
    for (const auto &scan_location : config["scan_locations"]) {
      args.scanner_positions.push_back(Eigen::Vector2d(scan_location["position"][0].as<double>(),
                                                       scan_location["position"][1].as<double>()));
      args.thetas.push_back(scan_location["theta"].as<double>());
    }
  }

  return args;
}

template Args<2> setup_from_yaml<2>(const sfs::path &config_path);
template Args<3> setup_from_yaml<3>(const sfs::path &config_path);
