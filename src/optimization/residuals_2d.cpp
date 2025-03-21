#include "residuals_2d.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <vector>

#include "map/map.hpp"
#include "utils.hpp"
#include "visualize/utils.hpp"

std::vector<double> compute_smoothness_residual_2d(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,  //
    const DerivativeType &type, const bool project_derivative) {
  if (type == DerivativeType::UPWIND) {
    return compute_smoothness_residual_2d_upwind(map, smoothness_factor, tree_global,
                                                 normals_global, project_derivative);
  } else if (type == DerivativeType::FORWARD) {
    return compute_smoothness_residual_2d_forward(map, smoothness_factor, tree_global,
                                                  normals_global, project_derivative);
  } else if (type == DerivativeType::CENTRAL) {
    return compute_smoothness_residual_2d_central(map, smoothness_factor, tree_global,
                                                  normals_global, project_derivative);
  } else {
    throw std::runtime_error("Invalid derivative type.");
  }
}

std::vector<double> compute_smoothness_residual_2d_upwind(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const bool project_derivative) {
  // Retrieve the derivative maps (for x and y).
  const std::array<Map<2>, 2> derivatives = map.df(DerivativeType::UPWIND);

  // Get grid dimensions.
  int num_x = derivatives[0].get_num_points(0);
  int num_y = derivatives[0].get_num_points(1);

  // Prepare a vector to hold one residual per grid point.
  std::vector<double> residuals;
  residuals.resize(map.total_points(), 0.0);

  // Loop over each grid point.
  for (int i = 0; i < num_x; i++) {
    for (int j = 0; j < num_y; j++) {
      // Extract the partial derivatives.
      const typename Map<2>::index_t index = {i, j};
      const double dDdx = derivatives[0].get_value_at(index);
      const double dDdy = derivatives[1].get_value_at(index);

      double grad_magnitude;
      if (project_derivative) {
        // Get the surface normal at the grid point.
        const typename Map<2>::Vector grid_pt = map.get_location(index);
        pcl::PointXY grid_pt_pcl;
        grid_pt_pcl.x = grid_pt.x();
        grid_pt_pcl.y = grid_pt.y();

        std::vector<int> nn_indices;
        std::vector<float> nn_dists;
        tree_global->nearestKSearch(grid_pt_pcl, 1, nn_indices, nn_dists);
        if (nn_indices.empty()) {
          continue;
        }
        const pcl::Normal n = normals_global->points[nn_indices[0]];

        // Normalize the scan normal.
        const double norm_val = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y);
        const double sn_x = (norm_val > 1e-6) ? n.normal_x / norm_val : 0.0;
        const double sn_y = (norm_val > 1e-6) ? n.normal_y / norm_val : 0.0;

        // Project the grid gradient onto the surface normal.
        grad_magnitude = dDdx * sn_x + dDdy * sn_y;
      } else {
        grad_magnitude = std::sqrt(dDdx * dDdx + dDdy * dDdy);
      }

      // Compute the residual for this grid point.
      const double point_residual = grad_magnitude - 1.0;

      // Scale the residual by the smoothness factor.
      residuals[i * num_y + j] = smoothness_factor * point_residual;
    }
  }

  return residuals;
}

std::vector<double> compute_smoothness_residual_2d_forward(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const bool project_derivative) {
  // Retrieve the derivative maps (for x and y).
  const std::array<Map<2>, 2> derivatives = map.df(DerivativeType::FORWARD);

  // Get grid dimensions.
  int num_x = derivatives[0].get_num_points(0);
  int num_y = derivatives[0].get_num_points(1);

  // Prepare a vector to hold one residual per grid point.
  std::vector<double> residuals;
  const uint residual_size = (map.get_num_points(0) - 1) * (map.get_num_points(1) - 1);
  residuals.resize(residual_size, 0.0);

  // Loop over each grid point.
  for (int i = 0; i < num_x - 1; i++) {
    for (int j = 0; j < num_y - 1; j++) {
      // Extract the partial derivatives.
      const typename Map<2>::index_t index = {i, j};
      const double dDdx = derivatives[0].get_value_at(index);
      const double dDdy = derivatives[1].get_value_at(index);

      double grad_magnitude;
      if (project_derivative) {
        // Get the surface normal at the grid point.
        const typename Map<2>::Vector grid_pt = map.get_location(index);
        pcl::PointXY grid_pt_pcl;
        grid_pt_pcl.x = grid_pt.x();
        grid_pt_pcl.y = grid_pt.y();

        std::vector<int> nn_indices;
        std::vector<float> nn_dists;
        tree_global->nearestKSearch(grid_pt_pcl, 1, nn_indices, nn_dists);
        if (nn_indices.empty()) {
          continue;
        }
        const pcl::Normal n = normals_global->points[nn_indices[0]];

        // Normalize the scan normal.
        const double norm_val = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y);
        const double sn_x = (norm_val > 1e-6) ? n.normal_x / norm_val : 0.0;
        const double sn_y = (norm_val > 1e-6) ? n.normal_y / norm_val : 0.0;

        // Project the grid gradient onto the surface normal.
        grad_magnitude = dDdx * sn_x + dDdy * sn_y;
      } else {
        grad_magnitude = std::sqrt(dDdx * dDdx + dDdy * dDdy);
      }

      // Compute the residual for this grid point.
      const double point_residual = grad_magnitude - 1.0;

      // Scale the residual by the smoothness factor.
      residuals[i * (num_y - 1) + j] = smoothness_factor * point_residual;
    }
  }

  return residuals;
}

std::vector<double> compute_smoothness_residual_2d_central(
    const Map<2> &map, const double smoothness_factor,
    pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals_global, const bool project_derivative) {
  // Retrieve the derivative maps (for x and y).
  const std::array<Map<2>, 2> derivatives = map.df(DerivativeType::CENTRAL);

  // Get grid dimensions.
  int num_x = derivatives[0].get_num_points(0);
  int num_y = derivatives[0].get_num_points(1);

  // Prepare a vector to hold one residual per grid point.
  std::vector<double> residuals;
  const uint residual_size = (map.get_num_points(0) - 1) * (map.get_num_points(1) - 1);
  residuals.resize(residual_size, 0.0);

  // Loop over each grid point.
  for (int i = 1; i < num_x - 1; i++) {
    for (int j = 1; j < num_y - 1; j++) {
      // Extract the partial derivatives.
      const typename Map<2>::index_t index = {i, j};
      const double dDdx = derivatives[0].get_value_at(index);
      const double dDdy = derivatives[1].get_value_at(index);

      double grad_magnitude;
      if (project_derivative) {
        // Get the surface normal at the grid point.
        const typename Map<2>::Vector grid_pt = map.get_location(index);
        pcl::PointXY grid_pt_pcl;
        grid_pt_pcl.x = grid_pt.x();
        grid_pt_pcl.y = grid_pt.y();

        std::vector<int> nn_indices;
        std::vector<float> nn_dists;
        tree_global->nearestKSearch(grid_pt_pcl, 1, nn_indices, nn_dists);
        if (nn_indices.empty()) {
          continue;
        }
        const pcl::Normal n = normals_global->points[nn_indices[0]];

        // Normalize the scan normal.
        const double norm_val = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y);
        const double sn_x = (norm_val > 1e-6) ? n.normal_x / norm_val : 0.0;
        const double sn_y = (norm_val > 1e-6) ? n.normal_y / norm_val : 0.0;

        // Project the grid gradient onto the surface normal.
        grad_magnitude = dDdx * sn_x + dDdy * sn_y;
      } else {
        grad_magnitude = std::sqrt(dDdx * dDdx + dDdy * dDdy);
      }

      // Compute the residual for this grid point.
      const double point_residual = grad_magnitude - 1.0;

      // Scale the residual by the smoothness factor.
      residuals[i * (num_y - 1) + j] = smoothness_factor * point_residual;
    }
  }

  return residuals;
}

Eigen::VectorXd compute_residuals_2d(const State<2> &state,
                                     const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds,
                                     const ObjectiveArgs &objective_args) {
  // Scan and line residuals
  const auto &point_value = generate_points_and_desired_values(state, point_clouds, objective_args);

  Eigen::VectorXd residuals;
  const uint smoothness_residuals =
      objective_args.smoothness_derivative_type == DerivativeType::FORWARD
          ? (state.map_.get_num_points(0) - 1) * (state.map_.get_num_points(1) - 1)
          : state.map_.total_points();

  residuals.resize(point_value.size() + smoothness_residuals);
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

  // Smoothing residuals
  // Global cloud
  const pcl::PointCloud<pcl::PointXY>::Ptr cloud_global =
      scans_to_global_pcl_2d(state.transformations_, point_clouds);
  pcl::search::KdTree<pcl::PointXY>::Ptr tree_global(new pcl::search::KdTree<pcl::PointXY>);
  tree_global->setInputCloud(cloud_global);

  // Global normals
  std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> scans = cloud_to_cloud_ptr(point_clouds);
  const pcl::PointCloud<pcl::Normal>::Ptr normals_global =
      compute_normals_global_2d(scans, state.transformations_);

  const std::vector<double> smoothing_residuals = compute_smoothness_residual_2d(
      state.map_, objective_args.smoothness_factor, tree_global, normals_global,
      objective_args.smoothness_derivative_type, objective_args.project_derivative);

  for (int i = 0; i < smoothing_residuals.size(); ++i) {
    residuals(point_value.size() + i) = smoothing_residuals[i];
  }

  return residuals;
}

double compute_roughness_residual_2d(const std::array<Map<2>, 2> &derivatives) {
  double roughness = 0;
  for (int i = 0; i < derivatives[0].get_num_points(0); i++) {
    for (int j = 0; j < derivatives[0].get_num_points(1); j++) {
      const typename Map<2>::index_t index = {i, j};
      const double dx = derivatives[0].get_value_at(index);
      const double dy = derivatives[1].get_value_at(index);

      const double norm = std::sqrt(dx * dx + dy * dy);
      const double norm_diff = std::abs(norm - 1);
      roughness += norm_diff;
    }
  }
  const double average_roughness = roughness / derivatives[0].total_points();
  return average_roughness;
}
