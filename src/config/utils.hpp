#pragma once
#include "map/utils.hpp"
#include "optimization/utils.hpp"
#include <filesystem>

template <int Dim> struct Args {
  MapArgs<Dim> map_args;
  ObjectiveArgs objective_args;
  OptimizationArgs optimization_args;
};

template <int Dim>
Args<Dim> setup_from_yaml(const std::filesystem::path &config_path);