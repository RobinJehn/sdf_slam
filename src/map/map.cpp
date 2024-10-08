#include "map.hpp"
#include "utils.hpp"

Map::Map(const int num_points, const float min_x, const float max_x,
         const float min_y, const float max_y, const float min_z,
         const float max_z)
    : dx_((max_x - min_x) / num_points), dy_((max_y - min_y) / num_points),
      dz_((max_z - min_z) / num_points), num_points_x_(num_points),
      num_points_y_(num_points), num_points_z_(num_points), min_x_(min_x),
      max_x_(max_x), min_y_(min_y), max_y_(max_y), min_z_(min_z),
      max_z_(max_z) {
  assert(min_x < max_x && min_y < max_y && min_z < max_z);

  // Initialize grid values to 0
  for (int x = 0; x < num_points_x_; ++x) {
    for (int y = 0; y < num_points_y_; ++y) {
      for (int z = 0; z < num_points_z_; ++z) {
        grid_values_[std::make_tuple(x, y, z)] = 0.0;
      }
    }
  }
}

Map::index_t Map::get_grid_coordinates(const Eigen::Vector3f &p) const {
  return std::make_tuple(static_cast<int>(floor(p.x() / dx_)),
                         static_cast<int>(floor(p.y() / dy_)),
                         static_cast<int>(floor(p.z() / dz_)));
}

float Map::get_value_at(const int x, const int y, const int z) const {
  return grid_values_.at(std::make_tuple(x, y, z));
}

void Map::set_value_at(const int x, const int y, const int z,
                       const float value) {
  grid_values_[std::make_tuple(x, y, z)] = value;
}

float Map::distance_to_surface(const Eigen::Vector3f &p) const {
  const auto [x_floor, y_floor, z_floor] = get_grid_coordinates(p);

  const float c000 = get_value_at(x_floor, y_floor, z_floor);
  const float c100 = get_value_at(x_floor + 1, y_floor, z_floor);
  const float c010 = get_value_at(x_floor, y_floor + 1, z_floor);
  const float c110 = get_value_at(x_floor + 1, y_floor + 1, z_floor);
  const float c001 = get_value_at(x_floor, y_floor, z_floor + 1);
  const float c101 = get_value_at(x_floor + 1, y_floor, z_floor + 1);
  const float c011 = get_value_at(x_floor, y_floor + 1, z_floor + 1);
  const float c111 = get_value_at(x_floor + 1, y_floor + 1, z_floor + 1);

  const Eigen::Vector3f p_floor(x_floor * dx_, y_floor * dy_, z_floor * dz_);

  return trilinear_interpolation(p, p_floor, dx_, dy_, dz_, c000, c100, c010,
                                 c110, c001, c101, c011, c111);
}
