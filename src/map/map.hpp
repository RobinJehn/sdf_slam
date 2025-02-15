#pragma once

#include <Eigen/Dense>
#include <functional>
#include <unordered_map>

#include "utils.hpp"

enum class DerivativeType { CENTRAL, UPWIND };

/**
 * @brief Generic class to represent a map in 2D or 3D. The map is represented
 * as a grid of points, where each point has a value. The map can be queried at
 * any point, and the value is interpolated from the grid points.
 *
 * @tparam Dim The dimension of the map (2 or 3)
 */
template <int Dim>
class Map {
  static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");

 public:
  using index_t = std::array<int, Dim>;
  using Vector = Eigen::Matrix<double, Dim, 1>;

  /**
   * @brief Get the value at a certain point in the map
   *
   * @param p
   *
   * @return double
   * @throws std::out_of_range if the point p is out of bounds
   */
  double value(const Vector &p) const;

  /**
   * @brief Construct a new Map object
   *
   * @param args Arguments to initialize the map
   */
  Map(const MapArgs<Dim> &args);

  int get_num_points(int dim) const { return num_points_[dim]; }
  std::array<int, Dim> get_num_points() const { return num_points_; }

  int total_points() const {
    int total_points = 1;
    for (int i = 0; i < Dim; ++i) {
      total_points *= num_points_[i];
    }
    return total_points;
  }

  Vector get_min_coords() const { return min_coords_; }
  double get_min_coord(int dim) const { return min_coords_[dim]; }

  Vector get_max_coords() const { return max_coords_; }
  double get_max_coord(int dim) const { return max_coords_[dim]; }

  /**
   * @brief Get the size of the map in a certain dimension
   *
   * @param dim The dimension
   */
  double get_d(int dim) const { return d_[dim]; }
  std::array<double, Dim> get_d() const { return d_; }

  void set_value_at(const index_t &index, const double value);

  /** Get the value at a certain grid point */
  double get_value_at(const index_t &index) const;

  /** @brief Get the minimum map value */
  double get_min_value() const;

  /** @brief Get the maximum map value */
  double get_max_value() const;

  /**
   * @brief Get the location of a certain grid point in global frame
   *
   * @param index The index of the grid point
   * @return The location of the grid point in global frame
   */
  Vector get_location(const index_t &index) const;

  /**
   * @brief Retrieves the neighboring indices of a given index in the map.
   *
   * Returns 2 * Dim neighbors for a given index in the map. Edges and corners
   * have fewer neighbors.
   *
   * @param index The index for which to find the neighbors.
   * @return the indices of the neighbors
   */
  std::vector<index_t> get_neighbours(const index_t &index) const;

  /**
   * @brief Check whether p is in bounds of the map
   *
   * @param p
   * @return
   */
  bool in_bounds(const Vector &p) const;

  /**
   * @brief Compute the derivative of the map
   * This function computes the derivative of the map in each dimension
   * using finite differences with neighboring nodes.
   *
   * @param type The type of derivative to compute
   *
   * @return std::array<Map<Dim>, Dim> A set of maps representing the derivative
   * in each dimension
   */
  std::array<Map<Dim>, Dim> df(const DerivativeType &type = DerivativeType::CENTRAL) const;

  /**
   * @brief Computes the grid indices for a given point in the map.
   *
   * @param p The point for which the grid indices are to be computed.
   *
   * @return The grid indices corresponding to the given point.
   * @throws std::out_of_range if the point p is out of bounds
   */
  index_t get_grid_indices(const Vector &p) const;

 private:
  /** Values at grid points */
  std::unordered_map<index_t, double, std::hash<index_t>> grid_values_;

  /** Size of the grid in each dimension */
  std::array<double, Dim> d_;

  /** Number of points in each dimension */
  std::array<int, Dim> num_points_;

  /** Minimum and maximum values in each dimension */
  Vector min_coords_;
  Vector max_coords_;

  /**
   * @brief Check whether point p is with in the bounds of the map.
   *
   * @param p
   * @throws std::out_of_range if the point p is out of bounds
   */
  void check_bounds(const Vector &p) const;
};

/** Custom hash function for index_t */
namespace std {
template <int Dim>
struct hash<std::array<int, Dim>> {
  size_t operator()(const std::array<int, Dim> &arr) const {
    std::hash<int> hasher;
    size_t seed = 0;
    for (int i = 0; i < Dim; ++i) {
      seed ^= hasher(arr[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
};  // namespace std
