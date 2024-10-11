#include "map.hpp"
#include "utils.hpp"
#include <iostream>

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
typename Map<Dim>::index_t Map<Dim>::get_grid_indices(const Vector &p) const {
  if constexpr (Dim == 2) {
    int x = static_cast<int>(floor((p.x() - min_coords_[0]) / d_[0]));
    int y = static_cast<int>(floor((p.y() - min_coords_[1]) / d_[1]));

    if (x < 0 || x >= num_points_[0]) {
      std::cerr << "Warning: Clamping x coordinate from " << x
                << " to range [0, " << num_points_[0] - 1 << "]\n";
      x = std::clamp(x, 0, num_points_[0] - 1);
    }
    if (y < 0 || y >= num_points_[1]) {
      std::cerr << "Warning: Clamping y coordinate from " << y
                << " to range [0, " << num_points_[1] - 1 << "]\n";
      y = std::clamp(y, 0, num_points_[1] - 1);
    }
    return {x, y};
  } else if constexpr (Dim == 3) {
    int x = static_cast<int>(floor((p.x() - min_coords_[0]) / d_[0]));
    int y = static_cast<int>(floor((p.y() - min_coords_[1]) / d_[1]));
    int z = static_cast<int>(floor((p.z() - min_coords_[2]) / d_[2]));

    if (x < 0 || x >= num_points_[0]) {
      std::cerr << "Warning: Clamping x coordinate from " << x
                << " to range [0, " << num_points_[0] - 1 << "]\n";
      x = std::clamp(x, 0, num_points_[0] - 1);
    }
    if (y < 0 || y >= num_points_[1]) {
      std::cerr << "Warning: Clamping y coordinate from " << y
                << " to range [0, " << num_points_[1] - 1 << "]\n";
      y = std::clamp(y, 0, num_points_[1] - 1);
    }
    if (z < 0 || z >= num_points_[2]) {
      std::cerr << "Warning: Clamping z coordinate from " << z
                << " to range [0, " << num_points_[2] - 1 << "]\n";
      z = std::clamp(z, 0, num_points_[2] - 1);
    }
    return {x, y, z};
  }
}

template <int Dim> double Map<Dim>::get_value_at(const index_t &coords) const {
  return grid_values_.at(coords);
}

template <int Dim>
void Map<Dim>::set_value_at(const index_t &coords, const double value) {
  grid_values_[coords] = value;
}

template <int Dim> double Map<Dim>::value(const Vector &p) const {
  const auto grid_indices = get_grid_indices(p);

  if constexpr (Dim == 2) {
    const int x_index = grid_indices[0];
    const int y_index = grid_indices[1];

    const double c00 = get_value_at({x_index, y_index});
    const double c10 = get_value_at({x_index + 1, y_index});
    const double c01 = get_value_at({x_index, y_index + 1});
    const double c11 = get_value_at({x_index + 1, y_index + 1});

    const Vector p_floor(x_index * d_[0], y_index * d_[1]);

    return bilinear_interpolation(p, p_floor, d_[0], d_[1], c00, c10, c01, c11);
  } else if constexpr (Dim == 3) {
    const int x_index = grid_indices[0];
    const int y_index = grid_indices[1];
    const int z_index = grid_indices[2];

    const double c000 = get_value_at({x_index, y_index, z_index});
    const double c100 = get_value_at({x_index + 1, y_index, z_index});
    const double c010 = get_value_at({x_index, y_index + 1, z_index});
    const double c110 = get_value_at({x_index + 1, y_index + 1, z_index});
    const double c001 = get_value_at({x_index, y_index, z_index + 1});
    const double c101 = get_value_at({x_index + 1, y_index, z_index + 1});
    const double c011 = get_value_at({x_index, y_index + 1, z_index + 1});
    const double c111 = get_value_at({x_index + 1, y_index + 1, z_index + 1});

    const Vector p_floor(x_index * d_[0], y_index * d_[1], z_index * d_[2]);

    return trilinear_interpolation(p, p_floor, d_[0], d_[1], d_[2], c000, c100,
                                   c010, c110, c001, c101, c011, c111);
  }
}

template <int Dim> std::array<Map<Dim>, Dim> Map<Dim>::df() const {
  // Initialize the derivative maps with the same grid setup as the original map
  std::array<Map<Dim>, Dim> derivatives = [this]() {
    if constexpr (Dim == 2) {
      return std::array<Map<Dim>, 2>{
          Map<Dim>(num_points_, min_coords_, max_coords_), // x-direction
          Map<Dim>(num_points_, min_coords_, max_coords_)  // y-direction
      };
    } else if constexpr (Dim == 3) {
      return std::array<Map<Dim>, 3>{
          Map<Dim>(num_points_, min_coords_, max_coords_), // x-direction
          Map<Dim>(num_points_, min_coords_, max_coords_), // y-direction
          Map<Dim>(num_points_, min_coords_, max_coords_)  // z-direction
      };
    }
  }();

  // Iterate over each node to compute its gradient
  for (int i = 0; i < num_points_[0]; ++i) {
    for (int j = 0; j < num_points_[1]; ++j) {
      if constexpr (Dim == 2) {
        double grad_x = 0.0f;
        double grad_y = 0.0f;

        // Compute gradient in the x-direction
        if (i > 0 && i < num_points_[0] - 1) {
          grad_x = (get_value_at({i + 1, j}) - get_value_at({i - 1, j})) /
                   (2.0 * d_[0]);
        } else if (i == 0) {
          grad_x = (get_value_at({i + 1, j}) - get_value_at({i, j})) / d_[0];
        } else if (i == num_points_[0] - 1) {
          grad_x = (get_value_at({i, j}) - get_value_at({i - 1, j})) / d_[0];
        }

        // Compute gradient in the y-direction
        if (j > 0 && j < num_points_[1] - 1) {
          grad_y = (get_value_at({i, j + 1}) - get_value_at({i, j - 1})) /
                   (2.0 * d_[1]);
        } else if (j == 0) {
          grad_y = (get_value_at({i, j + 1}) - get_value_at({i, j})) / d_[1];
        } else if (j == num_points_[1] - 1) {
          grad_y = (get_value_at({i, j}) - get_value_at({i, j - 1})) / d_[1];
        }

        // Set the computed gradients into the corresponding maps
        derivatives[0].set_value_at({i, j}, grad_x);
        derivatives[1].set_value_at({i, j}, grad_y);
      } else if constexpr (Dim == 3) {
        for (int k = 0; k < num_points_[2]; ++k) {
          double grad_x = 0.0f;
          double grad_y = 0.0f;
          double grad_z = 0.0f;

          // Compute gradient in the x-direction
          if (i > 0 && i < num_points_[0] - 1) {
            grad_x =
                (get_value_at({i + 1, j, k}) - get_value_at({i - 1, j, k})) /
                (2.0 * d_[0]);
          } else if (i == 0) {
            grad_x =
                (get_value_at({i + 1, j, k}) - get_value_at({i, j, k})) / d_[0];
          } else if (i == num_points_[0] - 1) {
            grad_x =
                (get_value_at({i, j, k}) - get_value_at({i - 1, j, k})) / d_[0];
          }

          // Compute gradient in the y-direction
          if (j > 0 && j < num_points_[1] - 1) {
            grad_y =
                (get_value_at({i, j + 1, k}) - get_value_at({i, j - 1, k})) /
                (2.0 * d_[1]);
          } else if (j == 0) {
            grad_y =
                (get_value_at({i, j + 1, k}) - get_value_at({i, j, k})) / d_[1];
          } else if (j == num_points_[1] - 1) {
            grad_y =
                (get_value_at({i, j, k}) - get_value_at({i, j - 1, k})) / d_[1];
          }

          // Compute gradient in the z-direction
          if (k > 0 && k < num_points_[2] - 1) {
            grad_z =
                (get_value_at({i, j, k + 1}) - get_value_at({i, j, k - 1})) /
                (2.0 * d_[2]);
          } else if (k == 0) {
            grad_z =
                (get_value_at({i, j, k + 1}) - get_value_at({i, j, k})) / d_[2];
          } else if (k == num_points_[2] - 1) {
            grad_z =
                (get_value_at({i, j, k}) - get_value_at({i, j, k - 1})) / d_[2];
          }

          // Set the computed gradients into the corresponding maps
          derivatives[0].set_value_at({i, j, k}, grad_x);
          derivatives[1].set_value_at({i, j, k}, grad_y);
          derivatives[2].set_value_at({i, j, k}, grad_z);
        }
      }
    }
  }

  return derivatives;
}

// Explicit template instantiation
template class Map<2>;
template class Map<3>;