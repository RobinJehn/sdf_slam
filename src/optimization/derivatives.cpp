#include "derivatives.hpp"

#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <array>

#include "map/map.hpp"
#include "utils.hpp"

void fill_dSmoothness_dMap_2d_upwind(const Map<2> &map, const double smoothness_factor,
                                     pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                                     const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                                     std::vector<Eigen::Triplet<double>> &triplet_list,
                                     const int residual_index_offset) {
  const int num_x = map.get_num_points(0);
  const int num_y = map.get_num_points(1);
  const int total_points = map.total_points();
  const std::array<int, 2> num_points = map.get_num_points();
  const double dx = map.get_d(0);
  const double dy = map.get_d(1);

  // Loop over each residual (1 per grid point)
  for (int i = 0; i < num_x; i++) {
    for (int j = 0; j < num_y; j++) {
      const int residual_index = residual_index_offset + i * num_y + j;
      const typename Map<2>::index_t index = {i, j};

      // Get the surface normal from the scan at this grid point.
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
      double norm_val = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y);
      double sn_x = (norm_val > 1e-6) ? n.normal_x / norm_val : 0.0;
      double sn_y = (norm_val > 1e-6) ? n.normal_y / norm_val : 0.0;

      // For each dimension, decide whether a forward or backward difference is used.
      double d_r_d_f_center = 0.0;  // Accumulated derivative contribution at f(i,j)

      // If we are inside the diff is flipped.
      const bool forward_diff_used_x =
          i == 0 || ((i < num_x - 1) && std::abs(map.get_value_at({i + 1, j})) <
                                            std::abs(map.get_value_at({i - 1, j})));
      const bool forward_diff_used_y =
          j == 0 || ((j < num_y - 1) && std::abs(map.get_value_at({i, j + 1})) <
                                            std::abs(map.get_value_at({i, j - 1})));
      const bool inside = forward_diff_used_x && map.get_value_at({i + 1, j}) < 0 ||
                          forward_diff_used_y && map.get_value_at({i, j + 1}) < 0 ||
                          !forward_diff_used_x && map.get_value_at({i - 1, j}) < 0 ||
                          !forward_diff_used_y && map.get_value_at({i, j - 1}) < 0;

      const int inside_factor = inside ? -1 : 1;

      // X-dimension:
      if (forward_diff_used_x) {
        const int flattened_index = map_index_to_flattened_index<2>(num_points, {i + 1, j});
        triplet_list.push_back({residual_index,                    //
                                /** parameter */ flattened_index,  //
                                /** value */ inside_factor * smoothness_factor * sn_x / dx});
        d_r_d_f_center += -inside_factor * smoothness_factor * sn_x / dx;
      } else {
        const int flattened_index = map_index_to_flattened_index<2>(num_points, {i - 1, j});
        triplet_list.push_back({residual_index,                    //
                                /** parameter */ flattened_index,  //
                                /** value */ -inside_factor * smoothness_factor * sn_x / dx});
        d_r_d_f_center += inside_factor * smoothness_factor * sn_x / dx;
      }

      // Y-dimension:
      if (forward_diff_used_y) {
        const int flattened_index = map_index_to_flattened_index<2>(num_points, {i, j + 1});
        triplet_list.push_back({residual_index,                    //
                                /** parameter */ flattened_index,  //
                                /** value */ inside_factor * smoothness_factor * sn_y / dy});
        d_r_d_f_center += -inside_factor * smoothness_factor * sn_y / dy;
      } else {
        const int flattened_index = map_index_to_flattened_index<2>(num_points, {i, j - 1});
        triplet_list.push_back({residual_index,                    //
                                /** parameter */ flattened_index,  //
                                /** value */ -inside_factor * smoothness_factor * sn_y / dy});
        d_r_d_f_center += inside_factor * smoothness_factor * sn_y / dy;
      }

      const int flattened_index = map_index_to_flattened_index<2>(num_points, index);
      triplet_list.push_back({residual_index,                    //
                              /** parameter */ flattened_index,  //
                              /** value */ d_r_d_f_center});
    }
  }
}

void fill_dSmoothness_dMap_2d_forward(const Map<2> &map, const double smoothness_factor,
                                      pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                                      const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                                      std::vector<Eigen::Triplet<double>> &triplet_list,
                                      const int residual_index_offset) {
  const int num_x = map.get_num_points(0);
  const int num_y = map.get_num_points(1);
  const int total_points = map.total_points();
  const std::array<int, 2> num_points = map.get_num_points();
  const double dx = map.get_d(0);
  const double dy = map.get_d(1);

  // Loop over each residual (1 per grid point)
  for (int i = 0; i < num_x; i++) {
    for (int j = 0; j < num_y; j++) {
      const int residual_index = residual_index_offset + i * num_y + j;
      const typename Map<2>::index_t index = {i, j};

      // Get the surface normal from the scan at this grid point.
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
      double norm_val = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y);
      double sn_x = (norm_val > 1e-6) ? n.normal_x / norm_val : 0.0;
      double sn_y = (norm_val > 1e-6) ? n.normal_y / norm_val : 0.0;

      // Accumulated derivative contribution at f(i,j)
      double d_r_d_f_center = 0.0;

      const bool inside = map.get_value_at({i + 1, j}) < 0 || map.get_value_at({i, j + 1}) < 0;

      const int inside_factor = inside ? -1 : 1;

      // X-dimension:
      int flattened_index = map_index_to_flattened_index<2>(num_points, {i + 1, j});
      triplet_list.push_back({residual_index,                    //
                              /** parameter */ flattened_index,  //
                              /** value */ inside_factor * smoothness_factor * sn_x / dx});
      d_r_d_f_center += -inside_factor * smoothness_factor * sn_x / dx;

      // Y-dimension:
      flattened_index = map_index_to_flattened_index<2>(num_points, {i, j + 1});
      triplet_list.push_back({residual_index,                    //
                              /** parameter */ flattened_index,  //
                              /** value */ inside_factor * smoothness_factor * sn_y / dy});
      d_r_d_f_center += -inside_factor * smoothness_factor * sn_y / dy;

      flattened_index = map_index_to_flattened_index<2>(num_points, index);
      triplet_list.push_back({residual_index,                    //
                              /** parameter */ flattened_index,  //
                              /** value */ d_r_d_f_center});
    }
  }
}

void fill_dSmoothness_dMap_2d(const Map<2> &map, const double smoothness_factor,
                              pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                              const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                              std::vector<Eigen::Triplet<double>> &triplet_list,
                              const int residual_index_offset, const DerivativeType type) {
  if (type == DerivativeType::UPWIND) {
    fill_dSmoothness_dMap_2d_upwind(map, smoothness_factor, tree_global, normals_global,
                                    triplet_list, residual_index_offset);
  } else if (type == DerivativeType::FORWARD) {
    fill_dSmoothness_dMap_2d_forward(map, smoothness_factor, tree_global, normals_global,
                                     triplet_list, residual_index_offset);
  } else {
    throw std::invalid_argument("Invalid derivative type");
  }
}
