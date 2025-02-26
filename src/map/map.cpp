#include "map.hpp"

#include <iostream>

#include "map_derivatives_2d.hpp"
#include "map_derivatives_3d.hpp"
#include "utils.hpp"

template <int Dim>
Map<Dim>::Map(const MapArgs<Dim> &args)
    : num_points_(args.num_points), min_coords_(args.min_coords), max_coords_(args.max_coords) {
  for (int i = 0; i < Dim; ++i) {
    d_[i] = (max_coords_[i] - min_coords_[i]) / (num_points_[i] - 1);
  }

  // Initialize grid values to 0
  if constexpr (Dim == 2) {
    for (int x = 0; x < num_points_[0]; ++x) {
      for (int y = 0; y < num_points_[1]; ++y) {
        grid_values_[{x, y}] = 0.0;
      }
    }
  } else if constexpr (Dim == 3) {
    for (int x = 0; x < num_points_[0]; ++x) {
      for (int y = 0; y < num_points_[1]; ++y) {
        for (int z = 0; z < num_points_[2]; ++z) {
          grid_values_[{x, y, z}] = 0.0;
        }
      }
    }
  }
}

template <int Dim>
bool Map<Dim>::in_bounds(const Vector &p) const {
  const bool x_inbounds = p.x() >= min_coords_[0] && p.x() <= max_coords_[0];
  const bool y_inbounds = p.y() >= min_coords_[1] && p.y() <= max_coords_[1];
  if constexpr (Dim == 2) {
    return x_inbounds && y_inbounds;
  } else if constexpr (Dim == 3) {
    const bool z_inbounds = p.z() >= min_coords_[2] && p.z() <= max_coords_[2];
    return x_inbounds && y_inbounds && z_inbounds;
  }
}

template <int Dim>
void Map<Dim>::check_bounds(const Vector &p) const {
  if (p.x() < min_coords_[0] || p.x() > max_coords_[0]) {
    throw std::out_of_range("Error: x coordinate " + std::to_string(p.x()) +
                            " is out of range. Valid range is [" + std::to_string(min_coords_[0]) +
                            ", " + std::to_string(max_coords_[0]) + "]");
  }
  if (p.y() < min_coords_[1] || p.y() > max_coords_[1]) {
    throw std::out_of_range("Error: y coordinate " + std::to_string(p.y()) +
                            " is out of range. Valid range is [" + std::to_string(min_coords_[1]) +
                            ", " + std::to_string(max_coords_[1]) + "]");
  }
  if constexpr (Dim == 3) {
    if (p.z() < min_coords_[2] || p.z() > max_coords_[2]) {
      throw std::out_of_range(
          "Error: z coordinate " + std::to_string(p.z()) + " is out of range. Valid range is [" +
          std::to_string(min_coords_[2]) + ", " + std::to_string(max_coords_[2]) + "]");
    }
  }
}

template <int Dim>
double Map<Dim>::get_min_value() const {
  double min_value = std::numeric_limits<double>::max();
  for (const auto &entry : grid_values_) {
    if (entry.second < min_value) {
      min_value = entry.second;
    }
  }
  return min_value;
}

template <int Dim>
double Map<Dim>::get_max_value() const {
  double max_value = std::numeric_limits<double>::min();
  for (const auto &entry : grid_values_) {
    if (entry.second > max_value) {
      max_value = entry.second;
    }
  }
  return max_value;
}

template <int Dim>
typename Map<Dim>::index_t Map<Dim>::get_grid_indices(const Vector &p) const {
  check_bounds(p);

  const int x = static_cast<int>(floor((p.x() - min_coords_[0]) / d_[0]));
  const int y = static_cast<int>(floor((p.y() - min_coords_[1]) / d_[1]));
  if constexpr (Dim == 2) {
    return {x, y};
  } else {
    const int z = static_cast<int>(floor((p.z() - min_coords_[2]) / d_[2]));
    return {x, y, z};
  }
}

template <int Dim>
double Map<Dim>::get_value_at(const index_t &index) const {
  return grid_values_.at(index);
}

template <int Dim>
void Map<Dim>::set_value_at(const index_t &index, const double value) {
  grid_values_[index] = value;
}

template <int Dim>
void Map<Dim>::set_value(const double value) {
  for (auto &entry : grid_values_) {
    entry.second = value;
  }
}

template <int Dim>
Map<Dim>::Vector Map<Dim>::get_location(const index_t &index) const {
  if constexpr (Dim == 2) {
    return Vector(index[0] * d_[0] + min_coords_.x(), index[1] * d_[1] + min_coords_.y());
  } else if constexpr (Dim == 3) {
    return Vector(index[0] * d_[0] + min_coords_.x(), index[1] * d_[1] + min_coords_.y(),
                  index[2] * d_[2] + min_coords_.z());
  }
}

template <int Dim>
double Map<Dim>::value(const Vector &p) const {
  const auto grid_indices = get_grid_indices(p);

  if constexpr (Dim == 2) {
    int x_index = grid_indices[0];
    int y_index = grid_indices[1];

    // Handle the case where the point is on the edge of the map.
    if (x_index == num_points_[0] - 1) {
      x_index = num_points_[0] - 2;
    }
    if (y_index == num_points_[1] - 1) {
      y_index = num_points_[1] - 2;
    }

    const double c00 = get_value_at({x_index, y_index});
    const double c10 = get_value_at({x_index + 1, y_index});
    const double c01 = get_value_at({x_index, y_index + 1});
    const double c11 = get_value_at({x_index + 1, y_index + 1});

    const Vector p_floor(x_index * d_[0] + min_coords_.x(), y_index * d_[1] + min_coords_.y());

    return bilinear_interpolation(p, p_floor, d_[0], d_[1], c00, c10, c01, c11);
  } else if constexpr (Dim == 3) {
    int x_index = grid_indices[0];
    int y_index = grid_indices[1];
    int z_index = grid_indices[2];

    // Handle the case where the point is on the edge of the map.
    if (x_index == num_points_[0] - 1) {
      x_index = num_points_[0] - 2;
    }
    if (y_index == num_points_[1] - 1) {
      y_index = num_points_[1] - 2;
    }
    if (z_index == num_points_[2] - 1) {
      z_index = num_points_[2] - 2;
    }

    const double c000 = get_value_at({x_index, y_index, z_index});
    const double c100 = get_value_at({x_index + 1, y_index, z_index});
    const double c010 = get_value_at({x_index, y_index + 1, z_index});
    const double c110 = get_value_at({x_index + 1, y_index + 1, z_index});
    const double c001 = get_value_at({x_index, y_index, z_index + 1});
    const double c101 = get_value_at({x_index + 1, y_index, z_index + 1});
    const double c011 = get_value_at({x_index, y_index + 1, z_index + 1});
    const double c111 = get_value_at({x_index + 1, y_index + 1, z_index + 1});

    const Vector p_floor(x_index * d_[0] + min_coords_.x(), y_index * d_[1] + min_coords_.y(),
                         z_index * d_[2] + min_coords_.z());

    return trilinear_interpolation(p, p_floor, d_[0], d_[1], d_[2], c000, c100, c010, c110, c001,
                                   c101, c011, c111);
  }
}

template <int Dim>
std::vector<typename Map<Dim>::index_t> Map<Dim>::get_neighbours(const index_t &index) const {
  std::vector<index_t> neighbours;

  // Handle edges and corners
  for (int i = 0; i < Dim; ++i) {
    if (index[i] > 0) {
      index_t neighbour = index;
      neighbour[i] -= 1;
      neighbours.push_back(neighbour);
    }
    if (index[i] < num_points_[i] - 1) {
      index_t neighbour = index;
      neighbour[i] += 1;
      neighbours.push_back(neighbour);
    }
  }

  return neighbours;
}

template <int Dim>
std::array<Map<Dim>, Dim> Map<Dim>::df(const DerivativeType &type) const {
  if constexpr (Dim == 3) {
    return df_3d(*this, type);
  } else if constexpr (Dim == 2) {
    return df_2d(*this, type);
  }
  static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
}

template <>
void Map<2>::from_ground_truth(const Scene &scene) {
  // Get the shapes from the scene.
  const auto &shapes = scene.get_shapes();

  // Loop over each grid index.
  for (int i = 0; i < num_points_[0]; ++i) {
    for (int j = 0; j < num_points_[1]; ++j) {
      std::array<int, 2> index = {i, j};
      // Get the global coordinates for this grid point.
      Eigen::Vector2d p = get_location(index);

      // Initialize min_distance to a large number.
      double min_distance = std::numeric_limits<double>::max();
      // For each shape, compute the (signed) distance.
      for (const auto &shape : shapes) {
        double d = shape->distance(p);
        // Choose the shape that minimizes the absolute distance.
        if (std::abs(d) < std::abs(min_distance)) {
          min_distance = d;
        }
      }
      // Set the computed (signed) distance at this grid point.
      set_value_at(index, min_distance);
    }
  }
}

// Explicit template instantiation
template class Map<2>;
template class Map<3>;
