#pragma once

#include <map>
#include <tuple>

#include <Eigen/Dense>

class Map {
  using index_t = std::tuple<int, int, int>;

public:
  float distance_to_surface(const Eigen::Vector3f &p) const;

  /**
   * @brief Construct a new Map object
   *
   * @param num_points Number of points in the grid for each dimension
   * @param min_x
   * @param max_x
   * @param min_y
   * @param max_y
   * @param min_z
   * @param max_z
   */
  Map(const int num_points, const float min_x, const float max_x,
      const float min_y, const float max_y, const float min_z,
      const float max_z);

  int get_num_points_x() const { return num_points_x_; }
  int get_num_points_y() const { return num_points_y_; }
  int get_num_points_z() const { return num_points_z_; }

  int get_num_points() const {
    return num_points_x_ * num_points_y_ * num_points_z_;
  }

  float get_min_x() const { return min_x_; }
  float get_max_x() const { return max_x_; }
  float get_min_y() const { return min_y_; }
  float get_max_y() const { return max_y_; }
  float get_min_z() const { return min_z_; }
  float get_max_z() const { return max_z_; }

  float get_dx() const { return dx_; }
  float get_dy() const { return dy_; }
  float get_dz() const { return dz_; }

  Map(const int num_points_x, const int num_points_y, const int num_points_z,
      const float min_x, const float max_x, const float min_y,
      const float max_y, const float min_z, const float max_z);

  Map(const float dx, const float dy, const float dz, const float min_x,
      const float max_x, const float min_y, const float max_y,
      const float min_z, const float max_z);

  void set_value_at(const int x, const int y, const int z, const float value);

  /** Get the value at a certain grid point */
  float get_value_at(const int x, const int y, const int z) const;

private:
  /** Distance values at grid points */
  std::map<index_t, float> grid_values_;

  /** Size of the grid in x, y and z direction */
  const float dx_;
  const float dy_;
  const float dz_;

  /** Number of points in each dimension */
  const int num_points_x_;
  const int num_points_y_;
  const int num_points_z_;

  /** Minimum and maximum values in each dimension */
  const float min_x_;
  const float max_x_;
  const float min_y_;
  const float max_y_;
  const float min_z_;
  const float max_z_;

  /** Get the grid coordinates of a certain point */
  index_t get_grid_coordinates(const Eigen::Vector3f &p) const;
};
