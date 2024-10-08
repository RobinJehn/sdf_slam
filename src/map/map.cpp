#include "map.hpp"
#include "utils.hpp"

template <int Dim>
Map<Dim>::Map(const std::array<int, Dim> &num_points, const Vector &min_coords,
              const Vector &max_coords)
    : num_points_(num_points), min_coords_(min_coords),
      max_coords_(max_coords) {
  for (int i = 0; i < Dim; ++i) {
    d_[i] = (max_coords[i] - min_coords[i]) / num_points[i];
  }

  // Initialize grid values to 0
  if constexpr (Dim == 2) {
    for (int x = 0; x < num_points_[0]; ++x) {
      for (int y = 0; y < num_points_[1]; ++y) {
        grid_values_[std::make_tuple(x, y)] = 0.0;
      }
    }
  } else if constexpr (Dim == 3) {
    for (int x = 0; x < num_points_[0]; ++x) {
      for (int y = 0; y < num_points_[1]; ++y) {
        for (int z = 0; z < num_points_[2]; ++z) {
          grid_values_[std::make_tuple(x, y, z)] = 0.0;
        }
      }
    }
  }
}

template <int Dim>
typename Map<Dim>::index_t
Map<Dim>::get_grid_coordinates(const Vector &p) const {
  if constexpr (Dim == 2) {
    return std::make_tuple(static_cast<int>(floor(p.x() / d_[0])),
                           static_cast<int>(floor(p.y() / d_[1])));
  } else if constexpr (Dim == 3) {
    return std::make_tuple(static_cast<int>(floor(p.x() / d_[0])),
                           static_cast<int>(floor(p.y() / d_[1])),
                           static_cast<int>(floor(p.z() / d_[2])));
  }
}

template <int Dim>
float Map<Dim>::get_value_at(const std::array<int, Dim> &coords) const {
  if constexpr (Dim == 2) {
    return grid_values_.at(std::make_tuple(coords[0], coords[1]));
  } else if constexpr (Dim == 3) {
    return grid_values_.at(std::make_tuple(coords[0], coords[1], coords[2]));
  }
}

template <int Dim>
void Map<Dim>::set_value_at(const std::array<int, Dim> &coords,
                            const float value) {
  if constexpr (Dim == 2) {
    grid_values_[std::make_tuple(coords[0], coords[1])] = value;
  } else if constexpr (Dim == 3) {
    grid_values_[std::make_tuple(coords[0], coords[1], coords[2])] = value;
  }
}

template <int Dim> float Map<Dim>::distance_to_surface(const Vector &p) const {
  const auto grid_coords = get_grid_coordinates(p);

  if constexpr (Dim == 2) {
    const int x_floor = std::get<0>(grid_coords);
    const int y_floor = std::get<1>(grid_coords);

    const float c00 = get_value_at({x_floor, y_floor});
    const float c10 = get_value_at({x_floor + 1, y_floor});
    const float c01 = get_value_at({x_floor, y_floor + 1});
    const float c11 = get_value_at({x_floor + 1, y_floor + 1});

    const Eigen::Vector2f p_floor(x_floor * d_[0], y_floor * d_[1]);

    return bilinear_interpolation(p, p_floor, d_[0], d_[1], c00, c10, c01, c11);
  } else if constexpr (Dim == 3) {
    const int x_floor = std::get<0>(grid_coords);
    const int y_floor = std::get<1>(grid_coords);
    const int z_floor = std::get<2>(grid_coords);

    const float c000 = get_value_at({x_floor, y_floor, z_floor});
    const float c100 = get_value_at({x_floor + 1, y_floor, z_floor});
    const float c010 = get_value_at({x_floor, y_floor + 1, z_floor});
    const float c110 = get_value_at({x_floor + 1, y_floor + 1, z_floor});
    const float c001 = get_value_at({x_floor, y_floor, z_floor + 1});
    const float c101 = get_value_at({x_floor + 1, y_floor, z_floor + 1});
    const float c011 = get_value_at({x_floor, y_floor + 1, z_floor + 1});
    const float c111 = get_value_at({x_floor + 1, y_floor + 1, z_floor + 1});

    const Eigen::Vector3f p_floor(x_floor * d_[0], y_floor * d_[1],
                                  z_floor * d_[2]);

    return trilinear_interpolation(p, p_floor, d_[0], d_[1], d_[2], c000, c100,
                                   c010, c110, c001, c101, c011, c111);
  }
}

// Explicit template instantiation
template class Map<2>;
template class Map<3>;