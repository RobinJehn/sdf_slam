#include "objective.hpp"
#include "state/state.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <vector>

template <int Dim>
ObjectiveFunctor<Dim>::ObjectiveFunctor(
    const int num_inputs, const int num_outputs,
    const std::array<int, Dim> &num_map_points, const Vector &min_coords,
    const Vector &max_coords,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size)
    : Functor<double>(num_inputs, num_outputs), point_clouds_(point_clouds),
      num_map_points_(num_map_points), min_coords_(min_coords),
      max_coords_(max_coords), number_of_points_(number_of_points),
      both_directions_(both_directions), step_size_(step_size) {}

template <int Dim>
int ObjectiveFunctor<Dim>::operator()(const Eigen::VectorXd &x,
                                      Eigen::VectorXd &fvec) const {
  State<Dim> state =
      unflatten<Dim>(x, num_map_points_, min_coords_, max_coords_);

  fvec = objective_vec<Dim>(state, point_clouds_, number_of_points_,
                            both_directions_, step_size_);

  return 0;
}

template <int Dim>
int ObjectiveFunctor<Dim>::df(const Eigen::VectorXd &x,
                              Eigen::MatrixXd &jacobian) const {
  State<Dim> state =
      unflatten<Dim>(x, num_map_points_, min_coords_, max_coords_);
  std::array<Map<Dim>, Dim> derivatives = state.map_.df();

  jacobian = Eigen::MatrixXd::Zero(values(), inputs());

  const auto &point_value = generate_points_and_desired_values(
      state, point_clouds_, number_of_points_, both_directions_, step_size_);
  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];

    // How the distance value changes if I change the map
    const auto &interpolation_point_indices =
        get_interpolation_point_indices<Dim>(point, state.map_);
    const auto &interpolation_weights =
        get_interpolation_weights<Dim>(point, state.map_);
    for (int j = 0; j < interpolation_point_indices.size(); ++j) {
      const auto &index = interpolation_point_indices[j];
      const int flattened_index =
          map_index_to_flattened_index<Dim>(num_map_points_, index);
      const double weight = interpolation_weights[j];
      jacobian(i, flattened_index) = weight;
    }

    // How the distance value changes if I change the point
    Eigen::Matrix<double, 1, Dim> dDF_dPoint;
    for (int d = 0; d < Dim; ++d) {
      dDF_dPoint[d] = derivatives[d].value(point);
    }

    const int points_per_transform = point_value.size() / point_clouds_.size();
    const int transformation_index = std::floor(i / points_per_transform);
    // How the point changes if I change the transformation
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

    // Fill the Jacobian matrix with the computed derivatives
    int total_map_points = 1;
    for (int d = 0; d < Dim; ++d) {
      total_map_points *= num_map_points_[d];
    }
    const int offset =
        total_map_points + transformation_index * (Dim + (Dim == 3 ? 3 : 1));
    for (int d = 0; d < Dim + (Dim == 3 ? 3 : 1); ++d) {
      jacobian(i, offset + d) = dDF_dTransformation(d);
    }
  }

  return 0;
}

// Explicit template instantiation
template class ObjectiveFunctor<2>;
template class ObjectiveFunctor<3>;