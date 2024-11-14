#include "utils.hpp"
#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

namespace sfs = std::filesystem;

template <int Dim> Args<Dim> setup_from_yaml(const sfs::path &config_path) {
  // Load the YAML file
  YAML::Node config = YAML::LoadFile(config_path.string());

  Args<Dim> args;

  // Read objective_args from YAML
  args.objective_args.scanline_points =
      config["objective_args"]["scanline_points"].as<int>();
  args.objective_args.step_size =
      config["objective_args"]["step_size"].as<double>();
  args.objective_args.both_directions =
      config["objective_args"]["both_directions"].as<bool>();
  args.objective_args.scan_line_factor =
      config["objective_args"]["scan_line_factor"].as<int>();
  args.objective_args.scan_point_factor =
      config["objective_args"]["scan_point_factor"].as<int>();

  // Read map_args from YAML
  if constexpr (Dim == 2) {
    args.map_args.num_points = {config["map_args"]["num_points"][0].as<int>(),
                                config["map_args"]["num_points"][1].as<int>()};
    args.map_args.min_coords =
        Eigen::Vector2d(config["map_args"]["min_coords"][0].as<double>(),
                        config["map_args"]["min_coords"][1].as<double>());
    args.map_args.max_coords =
        Eigen::Vector2d(config["map_args"]["max_coords"][0].as<double>(),
                        config["map_args"]["max_coords"][1].as<double>());
  } else if constexpr (Dim == 3) {
    args.map_args.num_points = {config["map_args"]["num_points"][0].as<int>(),
                                config["map_args"]["num_points"][1].as<int>(),
                                config["map_args"]["num_points"][2].as<int>()};
    args.map_args.min_coords =
        Eigen::Vector3d(config["map_args"]["min_coords"][0].as<double>(),
                        config["map_args"]["min_coords"][1].as<double>(),
                        config["map_args"]["min_coords"][2].as<double>());
    args.map_args.max_coords =
        Eigen::Vector3d(config["map_args"]["max_coords"][0].as<double>(),
                        config["map_args"]["max_coords"][1].as<double>(),
                        config["map_args"]["max_coords"][2].as<double>());
  }

  // Read optimization_args from YAML
  args.optimization_args.max_iters =
      config["optimization_args"]["max_iters"].as<int>();
  args.optimization_args.initial_lambda =
      config["optimization_args"]["initial_lambda"].as<double>();
  args.optimization_args.tolerance =
      config["optimization_args"]["tolerance"].as<double>();
  args.optimization_args.lambda_factor =
      config["optimization_args"]["lambda_factor"].as<double>();
  args.optimization_args.visualize =
      config["optimization_args"]["visualize"].as<bool>();
  args.optimization_args.std_out =
      config["optimization_args"]["std_out"].as<bool>();

  // Read general_args from YAML
  args.general_args.from_ground_truth =
      config["general_args"]["from_ground_truth"].as<bool>();
  args.general_args.data_path =
      sfs::path(config["general_args"]["data_path"].as<std::string>());

  if (args.objective_args.both_directions &&
      args.objective_args.scanline_points % 2 != 0) {
    throw std::runtime_error(
        "scanline_points must be even when both_directions is true.");
  }

  if (args.objective_args.scanline_points < 0) {
    throw std::runtime_error(
        "scanline_points must be greater than or equal to 0.");
  }

  if (args.objective_args.step_size <= 0) {
    throw std::runtime_error("step_size must be greater than 0.");
  }

  if (args.objective_args.scan_line_factor < 0) {
    throw std::runtime_error(
        "scan_line_factor must be greater than or equal to 0.");
  }

  if (args.objective_args.scan_point_factor < 0) {
    throw std::runtime_error(
        "scan_point_factor must be greater than or equal to 0.");
  }

  return args;
}

template Args<2> setup_from_yaml<2>(const sfs::path &config_path);
template Args<3> setup_from_yaml<3>(const sfs::path &config_path);