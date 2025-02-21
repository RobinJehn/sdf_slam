#include "map_derivatives_2d.hpp"

std::array<double, 2> upwind_difference_2d(const Map<2> &map, const Map<2>::index_t &index) {
  const auto num_points = map.get_num_points();

  if (index[0] < 0 || index[0] > num_points[0] - 1) {
    throw std::out_of_range("Index x out of bounds");
  }
  if (index[1] < 0 || index[1] > num_points[1] - 1) {
    throw std::out_of_range("Index y out of bounds");
  }
  if (num_points[0] < 2) {
    throw std::invalid_argument("Number of points in x-direction must be greater than 1");
  }
  if (num_points[1] < 2) {
    throw std::invalid_argument("Number of points in y-direction must be greater than 1");
  }

  std::array<double, 2> gradient;
  for (int dim = 0; dim < 2; ++dim) {
    if (index[dim] > 0 && index[dim] < num_points[dim] - 1) {
      auto index_lower = index;
      index_lower[dim] -= 1;
      const double value_lower_index = map.get_value_at(index_lower);

      const double own_value = map.get_value_at(index);

      auto index_higher = index;
      index_higher[dim] += 1;
      const double value_higher_index = map.get_value_at(index_higher);

      if (std::abs(value_higher_index) < std::abs(value_lower_index)) {
        gradient[dim] = (value_higher_index - own_value) / map.get_d(dim);
      } else {
        gradient[dim] = (own_value - value_lower_index) / map.get_d(dim);
      }
    } else if (index[dim] == 0) {  // Fallback to forward difference
      auto index_higher = index;
      index_higher[dim] += 1;
      const double value_higher_index = map.get_value_at(index_higher);

      const double own_value = map.get_value_at(index);

      gradient[dim] = (value_higher_index - own_value) / map.get_d(dim);
    } else if (index[dim] == num_points[dim] - 1) {  // Fallback to backward difference
      auto index_lower = index;
      index_lower[dim] -= 1;
      const double value_lower_index = map.get_value_at(index_lower);

      const double own_value = map.get_value_at(index);

      gradient[dim] = (own_value - value_lower_index) / map.get_d(dim);
    }
  }

  return gradient;
}

std::array<double, 2> central_difference_2d(const Map<2> &map, const Map<2>::index_t &index) {
  const auto num_points = map.get_num_points();

  if (index[0] < 0 || index[0] > num_points[0] - 1) {
    throw std::out_of_range("Index x out of bounds");
  }
  if (index[1] < 0 || index[1] > num_points[1] - 1) {
    throw std::out_of_range("Index y out of bounds");
  }
  if (num_points[0] < 2) {
    throw std::invalid_argument("Number of points in x-direction must be greater than 1");
  }
  if (num_points[1] < 2) {
    throw std::invalid_argument("Number of points in y-direction must be greater than 1");
  }

  std::array<double, 2> gradient;
  for (int dim = 0; dim < 2; ++dim) {
    if (index[dim] > 0 && index[dim] < num_points[dim] - 1) {
      auto index_lower = index;
      index_lower[dim] -= 1;
      const double value_lower_index = map.get_value_at(index_lower);

      auto index_higher = index;
      index_higher[dim] += 1;
      const double value_higher_index = map.get_value_at(index_higher);

      gradient[dim] = (value_higher_index - value_lower_index) / (2.0 * map.get_d(dim));
    } else if (index[dim] == 0) {  // Fallback to forward difference
      auto index_higher = index;
      index_higher[dim] += 1;
      const double value_higher_index = map.get_value_at(index_higher);

      const double own_value = map.get_value_at(index);

      gradient[dim] = (value_higher_index - own_value) / map.get_d(dim);
    } else if (index[dim] == num_points[dim] - 1) {  // Fallback to backward difference
      auto index_lower = index;
      index_lower[dim] -= 1;
      const double value_lower_index = map.get_value_at(index_lower);

      const double own_value = map.get_value_at(index);

      gradient[dim] = (own_value - value_lower_index) / map.get_d(dim);
    }
  }

  return gradient;
}

std::array<double, 2> forward_difference_2d(const Map<2> &map, const Map<2>::index_t &index) {
  const auto num_points = map.get_num_points();

  if (index[0] < 0 || index[0] > num_points[0] - 1) {
    throw std::out_of_range("Index x out of bounds");
  }
  if (index[1] < 0 || index[1] > num_points[1] - 1) {
    throw std::out_of_range("Index y out of bounds");
  }
  if (num_points[0] < 2) {
    throw std::invalid_argument("Number of points in x-direction must be greater than 1");
  }
  if (num_points[1] < 2) {
    throw std::invalid_argument("Number of points in y-direction must be greater than 1");
  }

  std::array<double, 2> gradient;
  for (int dim = 0; dim < 2; ++dim) {
    if (index[dim] == num_points[dim] - 1) {
      // Cannot compute forward difference at the last node
      gradient[dim] = 0.0;
      continue;
    }

    auto index_higher = index;
    index_higher[dim] += 1;
    const double value_higher_index = map.get_value_at(index_higher);

    const double own_value = map.get_value_at(index);

    gradient[dim] = (value_higher_index - own_value) / map.get_d(dim);
  }

  return gradient;
}

std::array<Map<2>, 2> df_2d(const Map<2> &map, const DerivativeType &type) {
  const auto num_points = map.get_num_points();
  MapArgs<2> args{num_points, map.get_min_coords(), map.get_max_coords()};
  std::array<Map<2>, 2> derivatives = {
      Map<2>(args),  // x-direction
      Map<2>(args),  // y-direction
  };

  // Iterate over each node to compute its gradient
  for (int i = 0; i < num_points[0]; ++i) {
    for (int j = 0; j < num_points[1]; ++j) {
      const Map<2>::index_t index = {i, j};

      std::array<double, 2> gradient;
      if (type == DerivativeType::CENTRAL) {
        gradient = central_difference_2d(map, index);
      } else if (type == DerivativeType::UPWIND) {
        gradient = upwind_difference_2d(map, index);
      } else if (type == DerivativeType::FORWARD) {
        gradient = forward_difference_2d(map, index);
      } else {
        throw std::invalid_argument("Invalid derivative type");
      }

      // Set the computed gradients into the corresponding maps
      for (int dim = 0; dim < 2; ++dim) {
        derivatives[dim].set_value_at(index, gradient[dim]);
      }
    }
  }

  return derivatives;
}
