#include "residuals_3d.hpp"

#include <array>

double compute_roughness_residual_3d(const std::array<Map<3>, 3> &derivatives) {
  double roughness = 0;
  for (int i = 0; i < derivatives[0].get_num_points(0); i++) {
    for (int j = 0; j < derivatives[0].get_num_points(1); j++) {
      for (int k = 0; k < derivatives[0].get_num_points(2); k++) {
        const typename Map<3>::index_t index = {i, j, k};
        const double dx = derivatives[0].get_value_at(index);
        const double dy = derivatives[1].get_value_at(index);
        const double dz = derivatives[2].get_value_at(index);

        const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double norm_diff = std::abs(norm - 1);
        roughness += norm_diff;
      }
    }
  }
  const double average_roughness = roughness / derivatives[0].total_points();
  return average_roughness;
}

Eigen::VectorXd compute_residuals_3d(
    const State<3> &state,  //
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds,
    const ObjectiveArgs &objective_args) {
  const auto &point_value = generate_points_and_desired_values(state, point_clouds, objective_args);

  Eigen::VectorXd residuals;
  residuals.resize(point_value.size() + 1);
  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];

    double interpolated_value = 0;
    if (state.map_.in_bounds(point)) {
      interpolated_value = state.map_.value(point);
    }

    double factor = i % (objective_args.scanline_points + 1) == 0 ? objective_args.scan_point_factor
                                                                  : objective_args.scan_line_factor;
    residuals(i) = factor * (interpolated_value - desired_value);
  }

  // Smoothing term
  double roughness = 0;
  const std::array<Map<3>, 3> derivatives = state.map_.df(DerivativeType::CENTRAL);
  const double average_roughness = compute_roughness_residual_3d(derivatives);

  residuals(point_value.size()) = objective_args.smoothness_factor * average_roughness;

  return residuals;
}
