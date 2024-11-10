#include "objective_ceres.hpp"
#include "state/state.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <vector>

// Constructor Implementation for ObjectiveFunctorCeres
template <int Dim>
ObjectiveFunctorCeres<Dim>::ObjectiveFunctorCeres(
    const std::array<int, Dim> &num_map_points, const Vector &min_coords,
    const Vector &max_coords,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size, const int num_inputs, const int num_outputs,
    const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame)
    : num_map_points_(num_map_points), min_coords_(min_coords),
      max_coords_(max_coords), point_clouds_(point_clouds),
      number_of_points_(number_of_points), both_directions_(both_directions),
      step_size_(step_size), num_inputs_(num_inputs), num_outputs_(num_outputs),
      initial_frame_(initial_frame) {}

// Compute residuals
template <int Dim>
bool ObjectiveFunctorCeres<Dim>::compute_residuals(
    const Eigen::VectorXd &x, Eigen::VectorXd &residuals) const {
  // Unflatten state from parameters
  State<Dim> state = unflatten<Dim>(x, num_map_points_, min_coords_,
                                    max_coords_, initial_frame_);

  // Compute residuals based on current state
  residuals = objective_vec<Dim>(state, point_clouds_, number_of_points_,
                                 both_directions_, step_size_);
  return true;
}

// Compute Jacobian
template <int Dim>
void ObjectiveFunctorCeres<Dim>::df(const Eigen::VectorXd &x,
                                    Eigen::MatrixXd &jacobian) const {
  // Unflatten state from parameters
  State<Dim> state = unflatten<Dim>(x, num_map_points_, min_coords_,
                                    max_coords_, initial_frame_);

  // Compute the Jacobian matrix using the derivatives of the map and
  // transformation
  // std::array<Map<Dim>, Dim> derivatives = state.map_.df();
  jacobian = Eigen::MatrixXd::Zero(num_outputs(), num_inputs());

  const auto &point_value = generate_points_and_desired_values<Dim>(
      state, point_clouds_, number_of_points_, both_directions_, step_size_);

  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];

    // How the residual changes w.r.t. the map
    const auto &interpolation_point_indices =
        get_interpolation_point_indices<Dim>(point, state.map_);
    const auto &interpolation_weights =
        get_interpolation_weights<Dim>(point, state.map_);
    for (int j = 0; j < interpolation_point_indices.size(); ++j) {
      const auto &index = interpolation_point_indices[j];
      const int flattened_index =
          map_index_to_flattened_index<Dim>(num_map_points_, index);
      jacobian(i, flattened_index) = interpolation_weights[j];
    }

    const int points_per_transform = point_value.size() / point_clouds_.size();
    const int transformation_index = std::floor(i / points_per_transform);
    if (transformation_index == 0) {
      continue;
    }

    // How the residual changes w.r.t. the transformation
    Eigen::Matrix<double, 1, Dim> dDF_dPoint =
        compute_analytical_derivative<Dim>(state.map_, point);

    Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)>
        dPoint_dTransformation = compute_transformation_derivative<Dim>(
            point, state.transformations_[transformation_index],
            /** numerical */ false);

    Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)> dDF_dTransformation =
        dDF_dPoint * dPoint_dTransformation;

    int total_map_points = 1;
    for (int d = 0; d < Dim; ++d) {
      total_map_points *= num_map_points_[d];
    }
    const int offset = total_map_points +
                       (transformation_index - 1) * (Dim + (Dim == 3 ? 3 : 1));
    for (int d = 0; d < Dim + (Dim == 3 ? 3 : 1); ++d) {
      jacobian(i, offset + d) = dDF_dTransformation(d);
    }
  }
}

template <int Dim> int ObjectiveFunctorCeres<Dim>::num_inputs() const {
  return num_inputs_;
}

template <int Dim> int ObjectiveFunctorCeres<Dim>::num_outputs() const {
  return num_outputs_;
}

ManualCostFunction::ManualCostFunction(ObjectiveFunctorCeres<2> *functor)
    : functor_(functor) {
  set_num_residuals(functor->num_outputs());
  mutable_parameter_block_sizes()->push_back(functor->num_inputs());
}

bool ManualCostFunction::Evaluate(double const *const *parameters,
                                  double *residuals, double **jacobians) const {
  Eigen::Map<const Eigen::VectorXd> params(parameters[0],
                                           functor_->num_inputs());

  // Compute residuals
  Eigen::VectorXd res;
  functor_->compute_residuals(params, res);
  for (int i = 0; i < res.size(); ++i) {
    residuals[i] = res[i];
  }

  if (jacobians != nullptr) {
    Eigen::MatrixXd jac;
    functor_->df(params, jac);
    for (int i = 0; i < jac.rows(); ++i) {
      for (int j = 0; j < jac.cols(); ++j) {
        jacobians[0][i * functor_->num_inputs() + j] = jac(i, j);
      }
    }
  }

  return true;
}

// Explicit template instantiation
template struct ObjectiveFunctorCeres<2>;
template struct ObjectiveFunctorCeres<3>;
