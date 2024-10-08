#pragma once
#include "map/map.hpp"
#include <Eigen/Dense>

struct State {
  const Map map_;
  const std::vector<Eigen::Affine3f> transformations_;

  State(const Map &map, const std::vector<Eigen::Affine3f> &transformations)
      : map_(map), transformations_(transformations) {}
};