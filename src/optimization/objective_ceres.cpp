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
    const double step_size)
    : num_map_points_(num_map_points), min_coords_(min_coords),
      max_coords_(max_coords), point_clouds_(point_clouds),
      number_of_points_(number_of_points), both_directions_(both_directions),
      step_size_(step_size) {}

// Compute residuals
template <int Dim>
bool ObjectiveFunctorCeres<Dim>::compute_residuals(
    const Eigen::VectorXd &x, Eigen::VectorXd &residuals) const {
  // Unflatten state from parameters
  State<Dim> state =
      unflatten<Dim>(x, num_map_points_, min_coords_, max_coords_);

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
  State<Dim> state =
      unflatten<Dim>(x, num_map_points_, min_coords_, max_coords_);

  // Compute the Jacobian matrix using the derivatives of the map and
  // transformation
  std::array<Map<Dim>, Dim> derivatives = state.map_.df();
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

    // How the residual changes w.r.t. the transformation
    Eigen::Matrix<double, 1, Dim> dDF_dPoint;
    for (int d = 0; d < Dim; ++d) {
      dDF_dPoint[d] = derivatives[d].value(point);
    }

    const int points_per_transform = point_value.size() / point_clouds_.size();
    const int transformation_index = std::floor(i / points_per_transform);

    Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)> dPoint_dTransformation;
    if constexpr (Dim == 2) {
      const Eigen::Matrix2d &rotation =
          state.transformations_[transformation_index].rotation();
      const double theta = std::atan2(rotation(1, 0), rotation(0, 0));
      dPoint_dTransformation =
          compute_transformation_derivative_2d(point, theta);
    } else if constexpr (Dim == 3) {
      const Eigen::Matrix3d &rotation =
          state.transformations_[transformation_index].rotation();
      const Eigen::Vector3d &euler_angles = rotation.eulerAngles(0, 1, 2);
      const double theta = euler_angles[0];
      const double phi = euler_angles[1];
      const double psi = euler_angles[2];
      dPoint_dTransformation =
          compute_transformation_derivative_3d(point, theta, phi, psi);
    }

    Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)> dDF_dTransformation =
        dDF_dPoint * dPoint_dTransformation;

    const int offset = num_map_points_.size() +
                       transformation_index * (Dim + (Dim == 3 ? 3 : 1));
    for (int d = 0; d < Dim + (Dim == 3 ? 3 : 1); ++d) {
      jacobian(i, offset + d) = dDF_dTransformation(d);
    }
  }
}

// Function to get number of input parameters
template <int Dim> int ObjectiveFunctorCeres<Dim>::num_inputs() const {
  return num_map_points_[0] * num_map_points_[1] +
         point_clouds_.size() * (Dim + (Dim == 3 ? 3 : 1));
}

// Function to get number of residuals (output size)
template <int Dim> int ObjectiveFunctorCeres<Dim>::num_outputs() const {
  return point_clouds_[0].size() * (num_line_points_ + 1) *
         point_clouds_.size();
}

// Explicit template instantiation
template struct ObjectiveFunctorCeres<2>;
template struct ObjectiveFunctorCeres<3>;
