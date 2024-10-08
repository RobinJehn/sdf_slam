#pragma once
#include "map/map.hpp"
#include <Eigen/Dense>
#include <vector>

template <int Dim> struct State {
  const Map<Dim> map_;
  const std::vector<Eigen::Transform<float, Dim, Eigen::Affine>>
      transformations_;

  State(const Map<Dim> &map,
        const std::vector<Eigen::Transform<float, Dim, Eigen::Affine>>
            &transformations);
};