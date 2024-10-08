#pragma once
#include "map/map.hpp"
#include <Eigen/Dense>

template <int Dim>
struct State {
  const Map<Dim> map_;
  const std::vector<Eigen::Transform<float, Dim, Eigen::Affine>> transformations_;

  State(const Map<Dim> &map, const std::vector<Eigen::Transform<float, Dim, Eigen::Affine>> &transformations)
      : map_(map), transformations_(transformations) {}
};