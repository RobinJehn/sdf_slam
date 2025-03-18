#include "objective.hpp"

#include <Eigen/Dense>
#include <vector>

#include "derivatives.hpp"
#include "residuals.hpp"
#include "state/state.hpp"
#include "utils.hpp"
#include "visualize/utils.hpp"

template <int Dim>
ObjectiveFunctor<Dim>::ObjectiveFunctor(
    const int num_inputs, const int num_outputs, const MapArgs<Dim> &map_args,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds,
    const ObjectiveArgs &objective_args,
    const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame)
    : Functor<double>(num_inputs, num_outputs),
      point_clouds_(point_clouds),
      map_args_(map_args),
      objective_args_(objective_args),
      initial_frame_(initial_frame) {}

template <int Dim>
int ObjectiveFunctor<Dim>::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
  State<Dim> state = unflatten<Dim>(x, initial_frame_, map_args_);

  fvec = compute_residuals<Dim>(state, point_clouds_, objective_args_);

  return 0;
}

template <int Dim>
std::pair<State<Dim>, std::array<Map<Dim>, Dim>>
ObjectiveFunctor<Dim>::compute_state_and_derivatives(const Eigen::VectorXd &x) const {
  State<Dim> state = unflatten<Dim>(x, initial_frame_, map_args_);
  std::array<Map<Dim>, Dim> derivatives = state.map_.df();
  return std::make_pair(state, derivatives);
}

template <int Dim>
int ObjectiveFunctor<Dim>::df(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobian) const {
  jacobian.setZero(values(), inputs());
  std::vector<Eigen::Triplet<double>> triplet_list = compute_jacobian_triplets(x);

  for (const auto &triplet : triplet_list) {
    jacobian(triplet.row(), triplet.col()) += triplet.value();
  }
  return 0;
}

template <int Dim>
std::vector<Eigen::Triplet<double>> ObjectiveFunctor<Dim>::compute_jacobian_triplets(
    const Eigen::VectorXd &x) const {
  std::vector<Eigen::Triplet<double>> triplet_list;

  // Needed variables
  const auto [state, derivatives] = compute_state_and_derivatives(x);
  const auto &point_value =
      generate_points_and_desired_values(state, point_clouds_, objective_args_);
  const double total_map_points = state.map_.total_points();

  // Point residuals
  for (size_t i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];
    if (!state.map_.in_bounds(point)) {
      continue;
    }

    const double residual_factor = (i % (objective_args_.scanline_points + 1) == 0)
                                       ? objective_args_.scan_point_factor
                                       : objective_args_.scan_line_factor;

    // dMap
    auto [interpolation_indices, interpolation_weights] =
        get_interpolation_values(point, state.map_);
    fill_dMap(triplet_list, i, residual_factor, interpolation_indices, interpolation_weights);

    // dTransform
    // Scan 0 is fixed
    // Recover the scan index. Scans are not necessarily the same size
    uint scan_index = 0;
    int a = 0;
    for (uint j = 0; j < point_clouds_.size(); ++j) {
      a += point_clouds_[j].size() * (objective_args_.scanline_points + 1);
      if (i < a) {
        scan_index = j;
        break;
      }
    }
    if (scan_index == 0) {
      continue;
    }

    const auto dResidual_dTransform =
        compute_dResidual_dTransform<Dim>(derivatives, point, state.transformations_[scan_index]);

    const uint offset = total_map_points + (scan_index - 1) * n_transformation_params_;
    fill_dTransform(triplet_list, i, residual_factor, offset, dResidual_dTransform);
  }

  // Smoothness residuals
  if constexpr (Dim == 2) {
    // Global cloud
    const pcl::PointCloud<pcl::PointXY>::Ptr cloud_global =
        scans_to_global_pcl_2d(state.transformations_, point_clouds_);
    pcl::search::KdTree<pcl::PointXY>::Ptr tree_global(new pcl::search::KdTree<pcl::PointXY>);
    tree_global->setInputCloud(cloud_global);

    // Global normals
    std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> scans = cloud_to_cloud_ptr(point_clouds_);
    const pcl::PointCloud<pcl::Normal>::Ptr normals_global =
        compute_normals_global_2d(scans, state.transformations_);

    fill_dSmoothness_dMap_2d(state.map_, objective_args_.smoothness_factor, tree_global,
                             normals_global, triplet_list, point_value.size(),
                             objective_args_.smoothness_derivative_type,
                             objective_args_.project_derivative);
  } else {
    fill_dRoughness_dMap(triplet_list, derivatives, objective_args_.smoothness_factor);
  }

  return triplet_list;
}

template <int Dim>
void ObjectiveFunctor<Dim>::fill_dMap(
    std::vector<Eigen::Triplet<double>> &triplet_list, const uint residual_index,
    const double residual_factor,
    const std::array<typename Map<Dim>::index_t, (1 << Dim)> &interpolation_indices,
    const Eigen::VectorXd &interpolation_weights) const {
  for (size_t j = 0; j < interpolation_indices.size(); ++j) {
    const auto &index = interpolation_indices[j];
    const int flattened_index = map_index_to_flattened_index<Dim>(map_args_.num_points, index);

    const double dMap = residual_factor * interpolation_weights[j];
    triplet_list.emplace_back(/* residual */ residual_index,  //
                              /* parameter */ flattened_index,
                              /* value */ dMap);
  }
}

template <int Dim>
void ObjectiveFunctor<Dim>::fill_dRoughness_dMap(std::vector<Eigen::Triplet<double>> &triplet_list,
                                                 const std::array<Map<Dim>, Dim> &map_derivatives,
                                                 const double factor) const {
  const std::vector<double> dRoughness_dMap = compute_dRoughness_dMap<Dim>(map_derivatives);
  for (size_t i = 0; i < dRoughness_dMap.size(); ++i) {
    triplet_list.emplace_back(/* residual */ values() - 1,  //
                              /* parameter */ i,
                              /* value */ factor * dRoughness_dMap[i]);
  }
}

template <int Dim>
void ObjectiveFunctor<Dim>::fill_dTransform(
    std::vector<Eigen::Triplet<double>> &triplet_list, const uint residual_index,
    const double residual_factor, const uint offset,
    const Eigen::Matrix<double, 1, n_transformation_params_> &dTransform) const {
  for (int d = 0; d < n_transformation_params_; ++d) {
    triplet_list.emplace_back(/* residual */ residual_index,  //
                              /* parameter */ offset + d,
                              /* value */ residual_factor * dTransform(d));
  }
}

template <int Dim>
int ObjectiveFunctor<Dim>::sparse_df(const Eigen::VectorXd &x,
                                     Eigen::SparseMatrix<double> &jacobian) const {
  const std::vector<Eigen::Triplet<double>> triplet_list = compute_jacobian_triplets(x);
  jacobian.resize(values(), inputs());
  jacobian.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return 0;
}

// Explicit template instantiation
template class ObjectiveFunctor<2>;
template class ObjectiveFunctor<3>;
