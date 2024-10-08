#include "objective.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

template <int Dim> Eigen::VectorXd flatten(const State<Dim> &state) {
  const int map_points = state.map_.get_num_points();
  const int num_transformations = state.transformations_.size();

  Eigen::VectorXd flattened(
      map_points + Dim * num_transformations +
      (Dim == 3 ? 3 * num_transformations : num_transformations));

  // Flatten the map
  int index = 0;
  for (int x = 0; x < state.map_.get_num_points(0); ++x) {
    for (int y = 0; y < state.map_.get_num_points(1); ++y) {
      if constexpr (Dim == 3) {
        for (int z = 0; z < state.map_.get_num_points(2); ++z) {
          flattened(index++) = state.map_.get_value_at({x, y, z});
        }
      } else if constexpr (Dim == 2) {
        flattened(index++) = state.map_.get_value_at({x, y});
      }
    }
  }

  // Flatten the transformations
  for (const auto &transform : state.transformations_) {
    const auto translation = transform.translation();

    Eigen::Matrix<float, 3, 1> euler_angles;
    if constexpr (Dim == 3) {
      euler_angles = transform.rotation().eulerAngles(0, 1, 2);
    } else if constexpr (Dim == 2) {
      euler_angles[0] =
          std::atan2(transform.rotation()(1, 0), transform.rotation()(0, 0));
    }

    for (int i = 0; i < Dim; ++i) {
      flattened(index++) = translation[i];
    }

    if constexpr (Dim == 3) {
      for (int i = 0; i < 3; ++i) {
        flattened(index++) = euler_angles[i];
      }
    } else if constexpr (Dim == 2) {
      flattened(index++) = euler_angles[0];
    }
  }

  return flattened;
}

template <int Dim>
State<Dim> unflatten(const Eigen::VectorXd &flattened_state,
                     const std::array<int, Dim> &num_points,
                     const Eigen::Matrix<float, Dim, 1> &min_coords,
                     const Eigen::Matrix<float, Dim, 1> &max_coords) {
  const int num_transformations =
      (flattened_state.size() - std::accumulate(num_points.begin(),
                                                num_points.end(), 1,
                                                std::multiplies<int>())) /
      (Dim + (Dim == 3 ? 3 : 1)); // 3 for Euler angles, 1 for angle

  Map<Dim> map(num_points, min_coords, max_coords);

  // Unflatten the map
  int index = 0;
  for (int x = 0; x < map.get_num_points(0); ++x) {
    for (int y = 0; y < map.get_num_points(1); ++y) {
      if constexpr (Dim == 3) {
        for (int z = 0; z < map.get_num_points(2); ++z) {
          map.set_value_at({x, y, z}, flattened_state(index++));
        }
      } else if constexpr (Dim == 2) {
        map.set_value_at({x, y}, flattened_state(index++));
      }
    }
  }

  // Unflatten the transformations
  std::vector<Eigen::Transform<float, Dim, Eigen::Affine>> transforms(
      num_transformations);
  for (auto &transform : transforms) {
    Eigen::Matrix<float, Dim, 1> translation;
    for (int i = 0; i < Dim; ++i) {
      translation[i] = flattened_state(index++);
    }

    if constexpr (Dim == 3) {
      Eigen::Matrix<float, 3, 1> euler_angles;
      for (int i = 0; i < 3; ++i) {
        euler_angles[i] = flattened_state(index++);
      }
      transform = Eigen::Translation<float, Dim>(translation) *
                  Eigen::AngleAxisf(euler_angles[0], Eigen::Vector3f::UnitX()) *
                  Eigen::AngleAxisf(euler_angles[1], Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(euler_angles[2], Eigen::Vector3f::UnitZ());
    } else if constexpr (Dim == 2) {
      float euler_angle = flattened_state(index++);
      transform = Eigen::Translation<float, Dim>(translation) *
                  Eigen::Rotation2Df(euler_angle);
    }
  }

  return State<Dim>(map, transforms);
}

template <int Dim>
ObjectiveFunctor<Dim>::ObjectiveFunctor(
    const int num_inputs, const int num_outputs,
    const std::array<int, Dim> &num_map_points, const Vector &min_coords,
    const Vector &max_coords,
    const std::vector<pcl::PointCloud<PointType>> &point_clouds)
    : Functor<double>(num_inputs, num_outputs), point_clouds_(point_clouds),
      num_map_points_(num_map_points), min_coords_(min_coords),
      max_coords_(max_coords) {}

template <int Dim>
int ObjectiveFunctor<Dim>::operator()(const Eigen::VectorXd &x,
                                      Eigen::VectorXd &fvec) const {
  State<Dim> state =
      unflatten<Dim>(x, num_map_points_, min_coords_, max_coords_);

  fvec = objective_vec<Dim>(state, point_clouds_);

  return 0;
}

template <int Dim>
Eigen::VectorXd objective_vec(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds) {
  assert(state.transformations_.size() == point_clouds.size() &&
         "Number of transformations must match number of point clouds");

  pcl::PointCloud<typename ObjectiveFunctor<Dim>::PointType> global_point_cloud;
  for (size_t i = 0; i < point_clouds.size(); ++i) {
    const auto &source_cloud = point_clouds[i];
    pcl::PointCloud<typename ObjectiveFunctor<Dim>::PointType>
        transformed_cloud;
    const Eigen::Transform<float, Dim, Eigen::Affine> &transform =
        state.transformations_[i];
    pcl::transformPointCloud(source_cloud, transformed_cloud, transform);

    global_point_cloud += transformed_cloud;
  }

  Eigen::VectorXd residuals(global_point_cloud.size());
  for (size_t i = 0; i < global_point_cloud.size(); ++i) {
    const auto &point = global_point_cloud[i];
    Eigen::Matrix<float, Dim, 1> point_vector;
    point_vector[0] = point.x;
    point_vector[1] = point.y;
    if constexpr (Dim == 3) {
      point_vector[2] = point.z;
    }
    residuals(i) = state.map_.distance_to_surface(point_vector);
  }

  return residuals;
}

template <int Dim>
float objective(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds) {
  assert(state.transformations_.size() == point_clouds.size() &&
         "Number of transformations must match number of point clouds");

  pcl::PointCloud<typename ObjectiveFunctor<Dim>::PointType> global_point_cloud;
  for (size_t i = 0; i < point_clouds.size(); ++i) {
    const auto &source_cloud = point_clouds[i];
    pcl::PointCloud<typename ObjectiveFunctor<Dim>::PointType>
        transformed_cloud;
    const Eigen::Transform<float, Dim, Eigen::Affine> &transform =
        state.transformations_[i];
    pcl::transformPointCloud(source_cloud, transformed_cloud, transform);

    global_point_cloud += transformed_cloud;
  }

  float distance = 0.0;
  for (const auto &point : global_point_cloud) {
    Eigen::Matrix<float, Dim, 1> point_vector;
    point_vector[0] = point.x;
    point_vector[1] = point.y;
    if constexpr (Dim == 3) {
      point_vector[2] = point.z;
    }
    distance += state.map_.distance_to_surface(point_vector);
  }

  return distance;
}

// Explicit template instantiation
template class ObjectiveFunctor<2>;
template class ObjectiveFunctor<3>;
template Eigen::VectorXd flatten<2>(const State<2> &state);
template Eigen::VectorXd flatten<3>(const State<3> &state);
template State<2> unflatten<2>(const Eigen::VectorXd &flattened_state,
                               const std::array<int, 2> &num_points,
                               const Eigen::Matrix<float, 2, 1> &min_coords,
                               const Eigen::Matrix<float, 2, 1> &max_coords);
template State<3> unflatten<3>(const Eigen::VectorXd &flattened_state,
                               const std::array<int, 3> &num_points,
                               const Eigen::Matrix<float, 3, 1> &min_coords,
                               const Eigen::Matrix<float, 3, 1> &max_coords);
template Eigen::VectorXd objective_vec<2>(
    const State<2> &state,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds);
template Eigen::VectorXd objective_vec<3>(
    const State<3> &state,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds);
template float
objective<2>(const State<2> &state,
             const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds);
template float
objective<3>(const State<3> &state,
             const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds);
