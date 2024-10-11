#include "objective.hpp"
#include "state/state.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
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
Eigen::VectorXd
objective_vec(const State<Dim> &state,
              const std::vector<pcl::PointCloud<typename std::conditional<
                  Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>> &point_clouds,
              const int number_of_points, const bool both_directions,
              const double step_size) {
  return compute_residuals(state, point_clouds, number_of_points,
                           both_directions, step_size);
}

template <int Dim>
std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
generate_points_and_desired_values(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size) {
  assert(state.transformations_.size() == point_clouds.size() &&
         "Number of transformations must match number of point clouds");

  std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
      point_desired_pairs;

  for (size_t i = 0; i < point_clouds.size(); ++i) {
    const auto &source_cloud = point_clouds[i];
    pcl::PointCloud<typename ObjectiveFunctor<Dim>::PointType>
        transformed_cloud;
    const auto &transform = state.transformations_[i];
    pcl::transformPointCloud(source_cloud, transformed_cloud,
                             transform.template cast<float>());

    for (const auto &point : transformed_cloud) {
      Eigen::Matrix<double, Dim, 1> point_vector;
      point_vector[0] = point.x;
      point_vector[1] = point.y;
      if constexpr (Dim == 3) {
        point_vector[2] = point.z;
      }

      // For point residuals, desired value is 0
      point_desired_pairs.emplace_back(point_vector, 0.0);

      if (number_of_points > 0) {
        Eigen::Matrix<double, Dim, 1> vector_to_origin =
            -point_vector.normalized() * step_size;
        int desired_points = number_of_points / (both_directions ? 2 : 1) + 1;
        for (int j = 1; j < desired_points; ++j) {
          Eigen::Matrix<double, Dim, 1> new_point_vector =
              point_vector + vector_to_origin * j;
          point_desired_pairs.emplace_back(new_point_vector, step_size * j);

          if (both_directions) {
            Eigen::Matrix<double, Dim, 1> new_point_vector_neg =
                point_vector - vector_to_origin * j;
            point_desired_pairs.emplace_back(new_point_vector_neg,
                                             -step_size * j);
          }
        }
      }
    }
  }

  return point_desired_pairs;
}

template <int Dim>
Eigen::VectorXd compute_residuals(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size) {
  const auto &point_value = generate_points_and_desired_values(
      state, point_clouds, number_of_points, both_directions, step_size);

  Eigen::VectorXd residuals(point_value.size());
  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];
    const double interpolated_value = state.map_.value(point);
    residuals(i) = desired_value - interpolated_value;
  }

  return residuals;
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
    const int offset = num_map_points_.size() +
                       transformation_index * (Dim + (Dim == 3 ? 3 : 1));
    for (int d = 0; d < Dim + (Dim == 3 ? 3 : 1); ++d) {
      jacobian(i, offset + d) = dDF_dTransformation(d);
    }
  }

  return 0;
}

template <int Dim>
double
ObjectiveFunctor<Dim>::compute_derivative(int residual_index, int param_index,
                                          const Eigen::VectorXd &x) const {

  return 0.0;
}

// Explicit template instantiation
template class ObjectiveFunctor<2>;
template class ObjectiveFunctor<3>;
template Eigen::VectorXd
objective_vec<2>(const State<2> &state,
                 const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds,
                 const int number_of_points, const bool both_directions,
                 const double step_size);
template Eigen::VectorXd objective_vec<3>(
    const State<3> &state,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds,
    const int number_of_points, const bool both_directions,
    const double step_size);