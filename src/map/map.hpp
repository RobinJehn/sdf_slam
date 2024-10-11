#pragma once

#include <Eigen/Dense>
#include <functional>
#include <tuple>
#include <unordered_map>

// Custom hash function for std::tuple
namespace std {
template <typename... T> struct hash<std::tuple<T...>> {
  size_t operator()(const std::tuple<T...> &t) const {
    return std::apply(
        [](const auto &...args) {
          return (... ^ std::hash<std::decay_t<decltype(args)>>{}(args));
        },
        t);
  }
};
} // namespace std

/**
 * @brief Generic class to represent a map in 2D or 3D. The map is represented
 * as a grid of points, where each point has a value. The map can be queried at
 * any point, and the value is interpolated from the grid points.
 *
 * @tparam Dim The dimension of the map (2 or 3)
 */
template <int Dim> class Map {
  static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");

  using index_t = std::conditional_t<Dim == 2, std::tuple<int, int>,
                                     std::tuple<int, int, int>>;
  using Vector = std::conditional_t<Dim == 2, Eigen::Vector2f, Eigen::Vector3f>;

public:
  /**
   * @brief Get the value at a certain point in the map
   *
   * @param p
   * @return float
   */
  float value(const Vector &p) const;

  /**
   * @brief Construct a new Map object
   *
   * @param num_points Number of points in the grid for each dimension
   * @param min_coords Minimum coordinates in each dimension
   * @param max_coords Maximum coordinates in each dimension
   */
  Map(const std::array<int, Dim> &num_points, const Vector &min_coords,
      const Vector &max_coords);

  int get_num_points(int dim) const { return num_points_[dim]; }

  int get_num_points() const {
    int total_points = 1;
    for (int i = 0; i < Dim; ++i) {
      total_points *= num_points_[i];
    }
    return total_points;
  }

  float get_min_coord(int dim) const { return min_coords_[dim]; }
  float get_max_coord(int dim) const { return max_coords_[dim]; }

  float get_d(int dim) const { return d_[dim]; }

  void set_value_at(const index_t &coords, const float value);

  /** Get the value at a certain grid point */
  float get_value_at(const index_t &coords) const;

  /**
   * @brief Compute the derivative of the map
   * This function computes the derivative of the map in each dimension
   * using finite differences with neighboring nodes.
   *
   * @return std::array<Map<Dim>, Dim> A set of maps representing the derivative
   * in each dimension
   */
  std::array<Map<Dim>, Dim> df() const;

private:
  /** Values at grid points */
  std::unordered_map<index_t, float> grid_values_;

  /** Size of the grid in each dimension */
  std::array<float, Dim> d_;

  /** Number of points in each dimension */
  std::array<int, Dim> num_points_;

  /** Minimum and maximum values in each dimension */
  Vector min_coords_;
  Vector max_coords_;

  /** Get the grid coordinates of a certain point */
  index_t get_grid_coordinates(const Vector &p) const;
};
