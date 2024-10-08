#include "objective.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

Eigen::VectorXd flatten(const State &state) {
  const int map_points = state.map_.get_num_points();
  const int num_transformations = state.transformations_.size();

  Eigen::VectorXd flattened(map_points + 6 * num_transformations);

  // Flatten the map
  int index = 0;
  for (int x = 0; x < state.map_.get_num_points_x(); ++x) {
    for (int y = 0; y < state.map_.get_num_points_y(); ++y) {
      for (int z = 0; z < state.map_.get_num_points_z(); ++z) {
        flattened(index++) = state.map_.get_value_at(x, y, z);
      }
    }
  }

  // Flatten the transformations
  for (const auto &transform : state.transformations_) {
    const Eigen::Vector3f translation = transform.translation();
    const Eigen::Vector3f euler_angles =
        transform.rotation().eulerAngles(0, 1, 2);

    flattened(index++) = translation.x();
    flattened(index++) = translation.y();
    flattened(index++) = translation.z();
    flattened(index++) = euler_angles.x();
    flattened(index++) = euler_angles.y();
    flattened(index++) = euler_angles.z();
  }

  return flattened;
}

State unflatten(const Eigen::VectorXd &flattened_state,
                const int num_map_points, const float min_x, const float max_x,
                const float min_y, const float max_y, const float min_z,
                const float max_z) {
  const int num_transformations = (flattened_state.size() - num_map_points) / 6;

  Map map(10, min_x, max_x, min_y, max_y, min_z, max_z);

  // Unflatten the map
  int index = 0;
  for (int x = 0; x < map.get_num_points_x(); ++x) {
    for (int y = 0; y < map.get_num_points_y(); ++y) {
      for (int z = 0; z < map.get_num_points_z(); ++z) {
        map.set_value_at(x, y, z, flattened_state(index++));
      }
    }
  }

  // Unflatten the transformations
  std::vector<Eigen::Affine3f> transforms(num_transformations);
  for (auto &transform : transforms) {
    Eigen::Vector3f translation;
    translation.x() = flattened_state(index++);
    translation.y() = flattened_state(index++);
    translation.z() = flattened_state(index++);

    Eigen::Vector3f euler_angles;
    euler_angles.x() = flattened_state(index++);
    euler_angles.y() = flattened_state(index++);
    euler_angles.z() = flattened_state(index++);

    transform = Eigen::Translation3f(translation) *
                Eigen::AngleAxisf(euler_angles.x(), Eigen::Vector3f::UnitX()) *
                Eigen::AngleAxisf(euler_angles.y(), Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(euler_angles.z(), Eigen::Vector3f::UnitZ());
  }

  return State(map, transforms);
}

ObjectiveFunctor::ObjectiveFunctor(
    const int num_inputs, const int num_outputs, const int num_map_points,
    const float min_x, const float max_x, const float min_y, const float max_y,
    const float min_z, const float max_z,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds)
    : Functor<double>(num_inputs, num_outputs), point_clouds_(point_clouds),
      num_map_points_(num_map_points), min_x_(min_x), max_x_(max_x),
      min_y_(min_y), max_y_(max_y), min_z_(min_z), max_z_(max_z) {}

int ObjectiveFunctor::operator()(const Eigen::VectorXd &x,
                                 Eigen::VectorXd &fvec) const {
  State state = unflatten(x, num_map_points_, min_x_, max_x_, min_y_, max_y_,
                          min_z_, max_z_);

  fvec = objective_vec(state, point_clouds_);

  return 0;
}

Eigen::VectorXd
objective_vec(const State &state,
              const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds) {
  assert(state.transformations_.size() == point_clouds.size() &&
         "Number of transformations must match number of point clouds");

  pcl::PointCloud<pcl::PointXYZ> global_point_cloud;
  for (size_t i = 0; i < point_clouds.size(); ++i) {
    const auto &source_cloud = point_clouds[i];
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    const Eigen::Affine3f &transform = state.transformations_[i];
    pcl::transformPointCloud(source_cloud, transformed_cloud, transform);

    global_point_cloud += transformed_cloud;
  }

  Eigen::VectorXd residuals(global_point_cloud.size());
  for (size_t i = 0; i < global_point_cloud.size(); ++i) {
    const auto &point = global_point_cloud[i];
    const Eigen::Vector3f point_vector(point.x, point.y, point.z);
    residuals(i) = state.map_.distance_to_surface(point_vector);
  }

  return residuals;
}

float objective(
    const State &state,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds) {
  assert(state.transformations_.size() == point_clouds.size() &&
         "Number of transformations must match number of point clouds");

  pcl::PointCloud<pcl::PointXYZ> global_point_cloud;
  for (size_t i = 0; i < point_clouds.size(); ++i) {
    const auto &source_cloud = point_clouds[i];
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    const Eigen::Affine3f &transform = state.transformations_[i];
    pcl::transformPointCloud(source_cloud, transformed_cloud, transform);

    global_point_cloud += transformed_cloud;
  }

  float distance = 0.0;
  for (const auto &point : global_point_cloud) {
    const Eigen::Vector3f point_vector(point.x, point.y, point.z);
    distance += state.map_.distance_to_surface(point_vector);
  }

  return distance;
}