#include "objective.hpp"
#include "state/state.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <vector>

template <int Dim>
ObjectiveFunctor<Dim>::ObjectiveFunctor(
    const int num_inputs, const int num_outputs, const MapArgs<Dim> &map_args,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds,
    const ObjectiveArgs &objective_args,
    const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame)
    : Functor<double>(num_inputs, num_outputs), point_clouds_(point_clouds),
      map_args_(map_args), objective_args_(objective_args),
      initial_frame_(initial_frame) {}

template <int Dim>
int ObjectiveFunctor<Dim>::operator()(const Eigen::VectorXd &x,
                                      Eigen::VectorXd &fvec) const {
  State<Dim> state = unflatten<Dim>(x, initial_frame_, map_args_);

  fvec = objective_vec<Dim>(state, point_clouds_, objective_args_);

  return 0;
}

template <int Dim>
std::pair<State<Dim>, std::array<Map<Dim>, Dim>>
ObjectiveFunctor<Dim>::compute_state_and_derivatives(
    const Eigen::VectorXd &x, const MapArgs<Dim> &map_args) const {
  State<Dim> state = unflatten<Dim>(x, initial_frame_, map_args);
  std::array<Map<Dim>, Dim> derivatives = state.map_.df();
  return std::make_pair(state, derivatives);
}

template <int Dim>
void ObjectiveFunctor<Dim>::fill_jacobian_dense(
    Eigen::MatrixXd &jacobian, const std::array<int, Dim> &num_map_points,
    int i, const Eigen::Matrix<double, 1, Dim> &dDF_dPoint,
    const Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)>
        &dPoint_dTransformation,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds,
    const std::array<typename Map<Dim>::index_t, (1 << Dim)>
        &interpolation_point_indices,
    const Eigen::VectorXd &interpolation_weights,
    const int transformation_index, const double residual_factor) const {
  for (int j = 0; j < interpolation_point_indices.size(); ++j) {
    const auto &index = interpolation_point_indices[j];
    const int flattened_index =
        map_index_to_flattened_index<Dim>(map_args_.num_points, index);
    const double weight = interpolation_weights[j];
    jacobian(i, flattened_index) = residual_factor * weight;
  }

  // Transformation 0 is fixed
  if (transformation_index == 0) {
    return;
  }

  int total_map_points = 1;
  for (int d = 0; d < Dim; ++d) {
    total_map_points *= num_map_points[d];
  }
  const int offset = total_map_points +
                     (transformation_index - 1) * (Dim + (Dim == 3 ? 3 : 1));

  Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)> dDF_dTransformation =
      dDF_dPoint * dPoint_dTransformation;
  for (int d = 0; d < Dim + (Dim == 3 ? 3 : 1); ++d) {
    jacobian(i, offset + d) = residual_factor * dDF_dTransformation(d);
  }
}

template <int Dim>
void ObjectiveFunctor<Dim>::fill_jacobian_sparse(
    std::vector<Eigen::Triplet<double>> &jacobian,
    const std::array<int, Dim> &num_map_points, int i,
    const Eigen::Matrix<double, 1, Dim> &dDF_dPoint,
    const Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)>
        &dPoint_dTransformation,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds,
    const std::array<typename Map<Dim>::index_t, (1 << Dim)>
        &interpolation_point_indices,
    const Eigen::VectorXd &interpolation_weights,
    const int transformation_index, const double residual_factor) const {
  for (int j = 0; j < interpolation_point_indices.size(); ++j) {
    const auto &index = interpolation_point_indices[j];
    const int flattened_index =
        map_index_to_flattened_index<Dim>(map_args_.num_points, index);
    const double weight = interpolation_weights[j];
    jacobian.emplace_back(i, flattened_index, residual_factor * weight);
  }

  // Transformation 0 is fixed
  if (transformation_index == 0) {
    return;
  }

  int total_map_points = 1;
  for (int d = 0; d < Dim; ++d) {
    total_map_points *= num_map_points[d];
  }

  const int offset = total_map_points +
                     (transformation_index - 1) * (Dim + (Dim == 3 ? 3 : 1));

  Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)> dDF_dTransformation =
      dDF_dPoint * dPoint_dTransformation;

  for (int d = 0; d < Dim + (Dim == 3 ? 3 : 1); ++d) {
    jacobian.emplace_back(i, offset + d,
                          residual_factor * dDF_dTransformation(d));
  }
}

template <int Dim>
int ObjectiveFunctor<Dim>::df(const Eigen::VectorXd &x,
                              Eigen::MatrixXd &jacobian) const {
  const auto [state, derivatives] = compute_state_and_derivatives(x, map_args_);
  jacobian.setZero(values(), inputs());

  const auto &point_value =
      generate_points_and_desired_values(state, point_clouds_, objective_args_);
  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];
    auto [interpolation_indices, interpolation_weights] =
        get_interpolation_values(point, state.map_);

    Eigen::Matrix<double, 1, Dim> dDF_dPoint =
        compute_approximate_derivative<Dim>(derivatives, point);

    double factor = i % (objective_args_.scanline_points + 1) == 0
                        ? objective_args_.scan_point_factor
                        : objective_args_.scan_line_factor;

    const int transformation_index =
        std::floor(i / (point_value.size() / point_clouds_.size()));
    auto dPoint_dTransformation = compute_transformation_derivative<Dim>(
        point, state.transformations_[transformation_index]);
    fill_jacobian_dense(jacobian, map_args_.num_points, i, dDF_dPoint,
                        dPoint_dTransformation, point_clouds_,
                        interpolation_indices, interpolation_weights,
                        transformation_index, factor);
  }
  return 0;
}

template <int Dim>
int ObjectiveFunctor<Dim>::sparse_df(
    const Eigen::VectorXd &x, Eigen::SparseMatrix<double> &jacobian) const {
  const auto [state, derivatives] = compute_state_and_derivatives(x, map_args_);
  std::vector<Eigen::Triplet<double>> tripletList;

  const auto &point_value =
      generate_points_and_desired_values(state, point_clouds_, objective_args_);
  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];
    if (!state.map_.in_bounds(point)) {
      continue;
    }
    auto [interpolation_indices, interpolation_weights] =
        get_interpolation_values(point, state.map_);

    Eigen::Matrix<double, 1, Dim> dDF_dPoint;
    for (int d = 0; d < Dim; ++d) {
      dDF_dPoint[d] =
          derivatives[d].in_bounds(point) ? derivatives[d].value(point) : 0;
    }

    double factor = i % (objective_args_.scanline_points + 1) == 0
                        ? objective_args_.scan_point_factor
                        : objective_args_.scan_line_factor;

    const int transformation_index =
        std::floor(i / (point_value.size() / point_clouds_.size()));
    auto dPoint_dTransformation = compute_transformation_derivative<Dim>(
        point, state.transformations_[transformation_index]);
    fill_jacobian_sparse(tripletList, map_args_.num_points, i, dDF_dPoint,
                         dPoint_dTransformation, point_clouds_,
                         interpolation_indices, interpolation_weights,
                         transformation_index, factor);
  }

  jacobian.resize(values(), inputs());
  jacobian.setFromTriplets(tripletList.begin(), tripletList.end());
  return 0;
}

// Explicit template instantiation
template class ObjectiveFunctor<2>;
template class ObjectiveFunctor<3>;