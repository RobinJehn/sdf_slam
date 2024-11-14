#include "utils.hpp"
#include "map/map.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <numeric>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <vector>

Eigen::Matrix<double, 3, 6>
compute_transformation_derivative_3d(const Eigen::Vector3d &p,
                                     const double theta, const double phi,
                                     const double psi) {
  // Derivative matrix (rotation and translation)
  Eigen::Matrix<double, 3, 6> derivative;

  // Translation derivatives
  derivative.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // Rotation derivatives
  derivative.col(3) = dR_dtheta(theta, phi, psi) * p;
  derivative.col(4) = dR_dphi(theta, phi, psi) * p;
  derivative.col(5) = dR_dpsi(theta, phi, psi) * p;

  return derivative;
}

Eigen::Matrix<double, 3, 6> compute_transformation_derivative_3d_numerical(
    const Eigen::Vector3d &p, const double theta, const double phi,
    const double psi) {
  Eigen::Matrix<double, 3, 6> derivative;

  // Translation derivatives
  derivative.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // Numerical approximation of rotation derivatives
  const double epsilon = 1e-6;

  auto numerical_derivative = [&](auto dR_func) {
    Eigen::Matrix3d R = dR_func(theta, phi, psi);
    Eigen::Matrix3d R_plus = dR_func(theta + epsilon, phi, psi);
    Eigen::Matrix3d R_minus = dR_func(theta - epsilon, phi, psi);
    return (R_plus - R_minus) / (2 * epsilon) * p;
  };

  derivative.col(3) = numerical_derivative(dR_dtheta);
  derivative.col(4) = numerical_derivative(dR_dphi);
  derivative.col(5) = numerical_derivative(dR_dpsi);

  return derivative;
}

Eigen::Matrix<double, 2, 3>
compute_transformation_derivative_2d(const Eigen::Vector2d &p,
                                     const double theta) {
  Eigen::Matrix<double, 2, 3> derivative;

  // Translation derivatives
  derivative.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();

  // Rotation derivative w.r.t. theta
  Eigen::Matrix2d dR_dtheta;
  dR_dtheta << -sin(theta), -cos(theta), cos(theta), -sin(theta);
  derivative.col(2) = dR_dtheta * p;

  return derivative;
}

Eigen::Matrix<double, 2, 3>
compute_transformation_derivative_2d_numerical(const Eigen::Vector2d &p,
                                               const double theta) {
  Eigen::Matrix<double, 2, 3> derivative;

  // Translation derivatives
  derivative.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();

  // Numerical approximation of rotation derivative w.r.t. theta
  const double epsilon = 1e-6;
  Eigen::Matrix2d R = Eigen::Rotation2Dd(theta).toRotationMatrix();
  Eigen::Matrix2d R_plus =
      Eigen::Rotation2Dd(theta + epsilon).toRotationMatrix();
  Eigen::Matrix2d R_minus =
      Eigen::Rotation2Dd(theta - epsilon).toRotationMatrix();
  Eigen::Matrix2d dR_dtheta_numerical = (R_plus - R_minus) / (2 * epsilon);

  derivative.col(2) = dR_dtheta_numerical * p;

  return derivative;
}

Eigen::Matrix3d dR_dtheta(const double theta, const double phi,
                          const double psi) {
  Eigen::Matrix3d dRz_dtheta;

  dRz_dtheta << -sin(theta), -cos(theta), 0, cos(theta), -sin(theta), 0, 0, 0,
      0;

  Eigen::Matrix3d R_y =
      Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Matrix3d R_x =
      Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitX()).toRotationMatrix();

  return dRz_dtheta * R_y * R_x;
}

Eigen::Matrix3d dR_dphi(const double theta, const double phi,
                        const double psi) {
  Eigen::Matrix3d dRy_dphi;

  dRy_dphi << -sin(phi), 0, cos(phi), 0, 0, 0, -cos(phi), 0, -sin(phi);

  Eigen::Matrix3d R_z =
      Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Matrix3d R_x =
      Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitX()).toRotationMatrix();

  return R_z * dRy_dphi * R_x;
}

Eigen::Matrix3d dR_dpsi(const double theta, const double phi,
                        const double psi) {
  Eigen::Matrix3d dRx_dpsi;

  dRx_dpsi << 0, 0, 0, 0, -sin(psi), -cos(psi), 0, cos(psi), -sin(psi);

  Eigen::Matrix3d R_z =
      Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Matrix3d R_y =
      Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()).toRotationMatrix();

  return R_z * R_y * dRx_dpsi;
}

template <int Dim> Eigen::VectorXd flatten(const State<Dim> &state) {
  const int map_points = state.map_.get_num_points();
  const int num_transformations = state.transformations_.size();

  Eigen::VectorXd flattened(
      map_points + Dim * (num_transformations - 1) +
      (Dim == 3 ? 3 * (num_transformations - 1) : (num_transformations - 1)));

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
  for (size_t i = 1; i < state.transformations_.size(); ++i) {
    const auto &transform = state.transformations_[i];
    const auto translation = transform.translation();

    Eigen::Matrix<double, Dim, 1> euler_angles;
    if constexpr (Dim == 3) {
      euler_angles = transform.rotation().eulerAngles(0, 1, 2);
    } else if constexpr (Dim == 2) {
      euler_angles[0] =
          std::atan2(transform.rotation()(1, 0), transform.rotation()(0, 0));
    }

    for (int j = 0; j < Dim; ++j) {
      flattened(index++) = translation[j];
    }

    if constexpr (Dim == 3) {
      for (int j = 0; j < 3; ++j) {
        flattened(index++) = euler_angles[j];
      }
    } else if constexpr (Dim == 2) {
      flattened(index++) = euler_angles[0];
    }
  }

  return flattened;
}

template <int Dim>
State<Dim>
unflatten(const Eigen::VectorXd &flattened_state,
          const Eigen::Transform<double, Dim, Eigen::Affine> &initial_frame,
          const MapArgs<Dim> &map_args) {
  Map<Dim> map(map_args);

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

  // Calculate the number of transformations
  const int num_transform_params =
      flattened_state.size() - map.get_num_points();
  const int num_params_per_transform = Dim + (Dim == 3 ? 3 : 1);
  const int num_transformations =
      num_transform_params / num_params_per_transform + 1;

  // Unflatten the transformations
  std::vector<Eigen::Transform<double, Dim, Eigen::Affine>> transforms(
      num_transformations);
  transforms[0] = initial_frame;
  for (size_t i = 1; i < transforms.size(); ++i) {
    Eigen::Matrix<double, Dim, 1> translation;
    for (int j = 0; j < Dim; ++j) {
      translation[j] = flattened_state(index++);
    }

    if constexpr (Dim == 3) {
      Eigen::Matrix<double, 3, 1> euler_angles;
      for (int j = 0; j < 3; ++j) {
        euler_angles[j] = flattened_state(index++);
      }
      transforms[i] =
          Eigen::Translation<double, Dim>(translation) *
          Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());
    } else if constexpr (Dim == 2) {
      double euler_angle = flattened_state(index++);
      transforms[i] = Eigen::Translation<double, Dim>(translation) *
                      Eigen::Rotation2Dd(euler_angle);
    }
  }

  return State<Dim>(map, transforms);
}

template <int Dim>
int map_index_to_flattened_index(const std::array<int, Dim> &num_points,
                                 const typename Map<Dim>::index_t &index) {
  int flattened_index = 0;
  int stride = 1;
  for (int i = Dim - 1; i >= 0; --i) {
    flattened_index += index[i] * stride;
    stride *= num_points[i];
  }
  return flattened_index;
}

template <int Dim>
std::array<typename Map<Dim>::index_t, (1 << Dim)>
get_interpolation_point_indices(const Eigen::Matrix<double, Dim, 1> &p,
                                const Map<Dim> &map) {
  const auto &index = map.get_grid_indices(p);

  std::array<typename Map<Dim>::index_t, (1 << Dim)>
      interpolation_point_indices;
  for (int i = 0; i < (1 << Dim); ++i) {
    typename Map<Dim>::index_t point;
    for (int d = 0; d < Dim; ++d) {
      point[d] = index[d] + ((i >> d) & 1);
    }
    interpolation_point_indices[i] = point;
  }

  return interpolation_point_indices;
}

template <int Dim>
Eigen::Matrix<double, (1 << Dim), 1>
get_interpolation_weights(const Eigen::Matrix<double, Dim, 1> &p,
                          const Map<Dim> &map) {
  const auto &index = map.get_grid_indices(p);

  Eigen::Matrix<double, Dim, 1> weights;
  for (int d = 0; d < Dim; ++d) {
    const double grid_size = map.get_d(d);
    const double floor_coord = index[d] * grid_size + map.get_min_coord(d);
    const double coord = p[d];

    weights[d] = (coord - floor_coord) / grid_size;
  }

  Eigen::Matrix<double, (1 << Dim), 1> interpolation_weights;
  for (int i = 0; i < (1 << Dim); ++i) {
    double weight = 1.0;
    for (int d = 0; d < Dim; ++d) {
      if ((i >> d) & 1) {
        weight *= weights[d];
      } else {
        weight *= (1.0 - weights[d]);
      }
    }
    interpolation_weights[i] = weight;
  }

  return interpolation_weights;
}

template <int Dim>
std::pair<std::array<typename Map<Dim>::index_t, (1 << Dim)>,
          Eigen::Matrix<double, (1 << Dim), 1>>
get_interpolation_values(const Eigen::Matrix<double, Dim, 1> &p,
                         const Map<Dim> &map) {

  auto interpolation_indices = get_interpolation_point_indices(p, map);
  auto interpolation_weights = get_interpolation_weights(p, map);

  return {interpolation_indices, interpolation_weights};
}

template <int Dim>
std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
generate_points_and_desired_values(
    const State<Dim> &state,
    const std::vector<pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &point_clouds,
    const ObjectiveArgs &objective_args) {
  assert(state.transformations_.size() == point_clouds.size() &&
         "Number of transformations must match number of point clouds");

  std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
      point_desired_pairs;

  for (size_t i = 0; i < point_clouds.size(); ++i) {
    const auto &scanner_cloud = point_clouds[i];

    pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>
        scanner_cloud_with_extra_points;
    std::vector<double> scanner_cloud_with_extra_points_values;

    for (const auto &point : scanner_cloud) {
      scanner_cloud_with_extra_points.push_back(point);
      scanner_cloud_with_extra_points_values.push_back(0.0);

      if (objective_args.scanline_points > 0) {
        Eigen::Matrix<double, Dim, 1> point_vector;
        point_vector[0] = point.x;
        point_vector[1] = point.y;
        if constexpr (Dim == 3) {
          point_vector[2] = point.z;
        }

        Eigen::Matrix<double, Dim, 1> vector_to_origin =
            -point_vector.normalized() * objective_args.step_size;
        int desired_points = objective_args.scanline_points /
                                 (objective_args.both_directions ? 2 : 1) +
                             1;
        for (int j = 1; j < desired_points; ++j) {
          Eigen::Matrix<double, Dim, 1> new_point_vector =
              point_vector + vector_to_origin * j;

          typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type
              point_plc;
          point_plc.x = new_point_vector[0];
          point_plc.y = new_point_vector[1];
          if constexpr (Dim == 3) {
            point_plc.z = new_point_vector[2];
          }
          scanner_cloud_with_extra_points.push_back(point_plc);
          scanner_cloud_with_extra_points_values.push_back(
              objective_args.step_size * j);

          if (objective_args.both_directions) {
            Eigen::Matrix<double, Dim, 1> new_point_vector_neg =
                point_vector - vector_to_origin * j;

            typename std::conditional<Dim == 2, pcl::PointXY,
                                      pcl::PointXYZ>::type point_plc_neg;
            point_plc_neg.x = new_point_vector_neg[0];
            point_plc_neg.y = new_point_vector_neg[1];
            if constexpr (Dim == 3) {
              point_plc_neg.z = new_point_vector_neg[2];
            }
            scanner_cloud_with_extra_points.push_back(point_plc_neg);
            scanner_cloud_with_extra_points_values.push_back(
                -objective_args.step_size * j);
          }
        }
      }
    }

    pcl::PointCloud<
        typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>
        transformed_cloud;
    const auto &transform = state.transformations_[i];
    pcl::transformPointCloud(scanner_cloud_with_extra_points, transformed_cloud,
                             transform.template cast<float>());

    for (int i = 0; i < transformed_cloud.size(); ++i) {
      auto &point = transformed_cloud[i];
      Eigen::Matrix<double, Dim, 1> point_vector;
      point_vector[0] = point.x;
      point_vector[1] = point.y;
      if constexpr (Dim == 3) {
        point_vector[2] = point.z;
      }
      point_desired_pairs.emplace_back(
          point_vector, scanner_cloud_with_extra_points_values[i]);
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
    const ObjectiveArgs &objective_args) {
  const auto &point_value =
      generate_points_and_desired_values(state, point_clouds, objective_args);

  Eigen::VectorXd residuals(point_value.size());
  for (int i = 0; i < point_value.size(); ++i) {
    const auto &[point, desired_value] = point_value[i];

    double interpolated_value = 0;
    if (state.map_.in_bounds(point)) {
      interpolated_value = state.map_.value(point);
    }

    double factor = i % (objective_args.scanline_points + 1) == 0
                        ? objective_args.scan_point_factor
                        : objective_args.scan_line_factor;
    residuals(i) = factor * (interpolated_value - desired_value);
  }

  return residuals;
}

template <int Dim>
Eigen::Matrix<double, Dim, 1>
compute_analytical_derivative(const Map<Dim> &map,
                              const Eigen::Matrix<double, Dim, 1> &point) {
  const auto [interpolation_indices, interpolation_weights] =
      get_interpolation_values<Dim>(point, map);

  std::array<double, Dim> d_values;
  for (int i = 0; i < Dim; ++i) {
    d_values[i] = map.get_d(i);
  }

  Eigen::Matrix<double, Dim, 1> dDF_dPoint;
  if constexpr (Dim == 2) {
    // 2D case
    const double dx = d_values[0];
    const double dy = d_values[1];

    const double b = interpolation_weights[2] + interpolation_weights[3];
    dDF_dPoint(0) = ((1 - b) * (map.get_value_at(interpolation_indices[1]) -
                                map.get_value_at(interpolation_indices[0])) +
                     b * (map.get_value_at(interpolation_indices[3]) -
                          map.get_value_at(interpolation_indices[2]))) /
                    dx;

    const double a = interpolation_weights[1] + interpolation_weights[3];
    dDF_dPoint(1) = ((1 - a) * (map.get_value_at(interpolation_indices[2]) -
                                map.get_value_at(interpolation_indices[0])) +
                     a * (map.get_value_at(interpolation_indices[3]) -
                          map.get_value_at(interpolation_indices[1]))) /
                    dy;
  } else if constexpr (Dim == 3) {
    // 3D case
    const double dx = d_values[0];
    const double dy = d_values[1];
    const double dz = d_values[2];

    const double c = interpolation_weights[4] + interpolation_weights[5] +
                     interpolation_weights[6] + interpolation_weights[7];
    dDF_dPoint(0) = ((1 - c) * (map.get_value_at(interpolation_indices[1]) -
                                map.get_value_at(interpolation_indices[0])) +
                     c * (map.get_value_at(interpolation_indices[5]) -
                          map.get_value_at(interpolation_indices[4]))) /
                    dx;

    const double b = interpolation_weights[2] + interpolation_weights[3] +
                     interpolation_weights[6] + interpolation_weights[7];
    dDF_dPoint(1) = ((1 - b) * (map.get_value_at(interpolation_indices[2]) -
                                map.get_value_at(interpolation_indices[0])) +
                     b * (map.get_value_at(interpolation_indices[6]) -
                          map.get_value_at(interpolation_indices[4]))) /
                    dy;

    const double a = interpolation_weights[1] + interpolation_weights[3] +
                     interpolation_weights[5] + interpolation_weights[7];
    dDF_dPoint(2) = ((1 - a) * (map.get_value_at(interpolation_indices[4]) -
                                map.get_value_at(interpolation_indices[0])) +
                     a * (map.get_value_at(interpolation_indices[7]) -
                          map.get_value_at(interpolation_indices[5]))) /
                    dz;
  }

  return dDF_dPoint;
}

template <int Dim>
Eigen::Matrix<double, Dim, 1>
compute_approximate_derivative(const std::array<Map<Dim>, Dim> &derivatives,
                               const Eigen::Matrix<double, Dim, 1> &point) {
  Eigen::Matrix<double, 1, Dim> dDF_dPoint;
  for (int d = 0; d < Dim; ++d) {
    if (derivatives[d].in_bounds(point)) {
      dDF_dPoint[d] = derivatives[d].value(point);
    } else {
      dDF_dPoint[d] = 0;
    }
  }

  return dDF_dPoint;
}

template <int Dim>
Eigen::Matrix<double, Dim, Dim + (Dim == 3 ? 3 : 1)>
compute_transformation_derivative(
    const Eigen::Matrix<double, Dim, 1> &point,
    const Eigen::Transform<double, Dim, Eigen::Affine> &transform,
    const bool numerical) {
  Eigen::Matrix<double, Dim, 1> point_in_scanner_frame =
      transform.inverse() * point;

  if constexpr (Dim == 2) {
    const Eigen::Matrix2d &rotation = transform.rotation();
    double theta = std::atan2(rotation(1, 0), rotation(0, 0));
    if (numerical) {
      return compute_transformation_derivative_2d_numerical(
          point_in_scanner_frame, theta);
    }
    return compute_transformation_derivative_2d(point_in_scanner_frame, theta);
  } else if constexpr (Dim == 3) {
    const Eigen::Matrix3d &rotation = transform.rotation();
    Eigen::Vector3d euler_angles = rotation.eulerAngles(0, 1, 2);
    if (numerical) {
      return compute_transformation_derivative_3d_numerical(
          point_in_scanner_frame, euler_angles[0], euler_angles[1],
          euler_angles[2]);
    }
    return compute_transformation_derivative_3d(
        point_in_scanner_frame, euler_angles[0], euler_angles[1],
        euler_angles[2]);
  }
}

template <int Dim>
Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)>
compute_derivative_map_value_wrt_transformation_numerical(
    const State<Dim> &state, const Eigen::Matrix<double, Dim, 1> &point,
    const Eigen::Transform<double, Dim, Eigen::Affine> &transform,
    const double epsilon) {
  Eigen::Matrix<double, 1, Dim + (Dim == 3 ? 3 : 1)> derivative;

  Eigen::Matrix<double, Dim, 1> point_in_scanner_frame =
      transform.inverse() * point;

  // Compute the numerical derivatives with respect to translation
  for (int i = 0; i < Dim; ++i) {
    Eigen::Transform<double, Dim, Eigen::Affine> transform_plus = transform;
    Eigen::Transform<double, Dim, Eigen::Affine> transform_minus = transform;
    transform_plus.translation()[i] += epsilon;
    transform_minus.translation()[i] -= epsilon;

    Eigen::Matrix<double, Dim, 1> point_plus =
        transform_plus * point_in_scanner_frame;
    Eigen::Matrix<double, Dim, 1> point_minus =
        transform_minus * point_in_scanner_frame;

    if (!state.map_.in_bounds(point_plus) ||
        !state.map_.in_bounds(point_minus)) {
      derivative(0, i) = 0;
      continue;
    }
    const double value_plus = state.map_.value(point_plus);
    const double value_minus = state.map_.value(point_minus);
    derivative(0, i) = (value_plus - value_minus) / (2 * epsilon);
  }

  // Compute the numerical derivatives with respect to rotation
  if constexpr (Dim == 2) {
    const Eigen::Matrix2d rotation = transform.rotation();
    const double theta = std::atan2(rotation(1, 0), rotation(0, 0));

    Eigen::Transform<double, Dim, Eigen::Affine> transform_plus = transform;
    Eigen::Transform<double, Dim, Eigen::Affine> transform_minus = transform;
    transform_plus.rotate(Eigen::Rotation2Dd(epsilon));
    transform_minus.rotate(Eigen::Rotation2Dd(-epsilon));

    Eigen::Matrix<double, Dim, 1> point_plus =
        transform_plus * point_in_scanner_frame;
    Eigen::Matrix<double, Dim, 1> point_minus =
        transform_minus * point_in_scanner_frame;

    if (!state.map_.in_bounds(point_plus) ||
        !state.map_.in_bounds(point_minus)) {
      derivative(0, 2) = 0;
      return derivative;
    }

    const double value_plus = state.map_.value(point_plus);
    const double value_minus = state.map_.value(point_minus);
    derivative(0, 2) = (value_plus - value_minus) / (2 * epsilon);
  } else if constexpr (Dim == 3) {
    const Eigen::Matrix3d rotation = transform.rotation();
    Eigen::Vector3d euler_angles = rotation.eulerAngles(0, 1, 2);

    for (int i = 0; i < 3; ++i) {
      Eigen::Transform<double, Dim, Eigen::Affine> transform_plus = transform;
      Eigen::Transform<double, Dim, Eigen::Affine> transform_minus = transform;

      Eigen::Vector3d euler_angles_plus = euler_angles;
      euler_angles_plus[i] += epsilon;
      Eigen::Vector3d euler_angles_minus = euler_angles;
      euler_angles_minus[i] -= epsilon;

      transform_plus.rotate(
          Eigen::AngleAxisd(euler_angles_plus[0], Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(euler_angles_plus[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(euler_angles_plus[2], Eigen::Vector3d::UnitZ()));
      transform_minus.rotate(
          Eigen::AngleAxisd(euler_angles_minus[0], Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(euler_angles_minus[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(euler_angles_minus[2], Eigen::Vector3d::UnitZ()));

      Eigen::Matrix<double, Dim, 1> point_plus =
          transform_plus * point_in_scanner_frame;
      Eigen::Matrix<double, Dim, 1> point_minus =
          transform_minus * point_in_scanner_frame;

      if (!state.map_.in_bounds(point_plus) ||
          !state.map_.in_bounds(point_minus)) {
        derivative(0, Dim + i) = 0;
        continue;
      }

      const double value_plus = state.map_.value(point_plus);
      const double value_minus = state.map_.value(point_minus);
      derivative(0, Dim + i) = (value_plus - value_minus) / (2 * epsilon);
    }
  }

  return derivative;
}

// Explicit template instantiation
template Eigen::VectorXd flatten<2>(const State<2> &state);
template Eigen::VectorXd flatten<3>(const State<3> &state);
template State<2>
unflatten<2>(const Eigen::VectorXd &flattened_state,
             const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
             const MapArgs<2> &map_args);
template State<3>
unflatten<3>(const Eigen::VectorXd &flattened_state,
             const Eigen::Transform<double, 3, Eigen::Affine> &initial_frame,
             const MapArgs<3> &map_args);
template int
map_index_to_flattened_index<2>(const std::array<int, 2> &num_points,
                                const typename Map<2>::index_t &index);
template int
map_index_to_flattened_index<3>(const std::array<int, 3> &num_points,
                                const typename Map<3>::index_t &index);
template std::array<typename Map<2>::index_t, (1 << 2)>
get_interpolation_point_indices<2>(const Eigen::Matrix<double, 2, 1> &p,
                                   const Map<2> &map);
template std::array<typename Map<3>::index_t, (1 << 3)>
get_interpolation_point_indices<3>(const Eigen::Matrix<double, 3, 1> &p,
                                   const Map<3> &map);
template Eigen::Matrix<double, (1 << 2), 1>
get_interpolation_weights<2>(const Eigen::Matrix<double, 2, 1> &p,
                             const Map<2> &map);
template Eigen::Matrix<double, (1 << 3), 1>
get_interpolation_weights<3>(const Eigen::Matrix<double, 3, 1> &p,
                             const Map<3> &map);

template std::pair<std::array<typename Map<2>::index_t, (1 << 2)>,
                   Eigen::Matrix<double, (1 << 2), 1>>
get_interpolation_values<2>(const Eigen::Matrix<double, 2, 1> &p,
                            const Map<2> &map);
template std::pair<std::array<typename Map<3>::index_t, (1 << 3)>,
                   Eigen::Matrix<double, (1 << 3), 1>>
get_interpolation_values<3>(const Eigen::Matrix<double, 3, 1> &p,
                            const Map<3> &map);

template Eigen::Matrix<double, 2, 1>
compute_analytical_derivative<2>(const Map<2> &map,
                                 const Eigen::Matrix<double, 2, 1> &point);
template Eigen::Matrix<double, 3, 1>
compute_analytical_derivative<3>(const Map<3> &map,
                                 const Eigen::Matrix<double, 3, 1> &point);

template Eigen::Matrix<double, 2, 1>
compute_approximate_derivative<2>(const std::array<Map<2>, 2> &derivatives,
                                  const Eigen::Matrix<double, 2, 1> &point);
template Eigen::Matrix<double, 3, 1>
compute_approximate_derivative<3>(const std::array<Map<3>, 3> &derivatives,
                                  const Eigen::Matrix<double, 3, 1> &point);

template Eigen::Matrix<double, 2, 3> compute_transformation_derivative<2>(
    const Eigen::Matrix<double, 2, 1> &point,
    const Eigen::Transform<double, 2, Eigen::Affine> &transform,
    const bool numerical);
template Eigen::Matrix<double, 3, 6> compute_transformation_derivative<3>(
    const Eigen::Matrix<double, 3, 1> &point,
    const Eigen::Transform<double, 3, Eigen::Affine> &transform,
    const bool numerical);

template Eigen::Matrix<double, 1, 3>
compute_derivative_map_value_wrt_transformation_numerical<2>(
    const State<2> &state, const Eigen::Matrix<double, 2, 1> &point,
    const Eigen::Transform<double, 2, Eigen::Affine> &transform,
    const double epsilon);
template Eigen::Matrix<double, 1, 6>
compute_derivative_map_value_wrt_transformation_numerical<3>(
    const State<3> &state, const Eigen::Matrix<double, 3, 1> &point,
    const Eigen::Transform<double, 3, Eigen::Affine> &transform,
    const double epsilon);

template Eigen::VectorXd compute_residuals<2>(
    const State<2> &state,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &point_clouds,
    const ObjectiveArgs &objective_args);

template Eigen::VectorXd compute_residuals<3>(
    const State<3> &state,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &point_clouds,
    const ObjectiveArgs &objective_args);