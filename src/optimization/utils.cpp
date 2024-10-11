#include "utils.hpp"
#include "map/map.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <numeric>
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

    Eigen::Matrix<double, Dim, 1> euler_angles;
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
                     const Eigen::Matrix<double, Dim, 1> &min_coords,
                     const Eigen::Matrix<double, Dim, 1> &max_coords) {
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
  std::vector<Eigen::Transform<double, Dim, Eigen::Affine>> transforms(
      num_transformations);
  for (auto &transform : transforms) {
    Eigen::Matrix<double, Dim, 1> translation;
    for (int i = 0; i < Dim; ++i) {
      translation[i] = flattened_state(index++);
    }

    if constexpr (Dim == 3) {
      Eigen::Matrix<double, 3, 1> euler_angles;
      for (int i = 0; i < 3; ++i) {
        euler_angles[i] = flattened_state(index++);
      }
      transform = Eigen::Translation<double, Dim>(translation) *
                  Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitX()) *
                  Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());
    } else if constexpr (Dim == 2) {
      double euler_angle = flattened_state(index++);
      transform = Eigen::Translation<double, Dim>(translation) *
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

// Explicit template instantiation
template Eigen::VectorXd flatten<2>(const State<2> &state);
template Eigen::VectorXd flatten<3>(const State<3> &state);
template State<2> unflatten<2>(const Eigen::VectorXd &flattened_state,
                               const std::array<int, 2> &num_points,
                               const Eigen::Matrix<double, 2, 1> &min_coords,
                               const Eigen::Matrix<double, 2, 1> &max_coords);
template State<3> unflatten<3>(const Eigen::VectorXd &flattened_state,
                               const std::array<int, 3> &num_points,
                               const Eigen::Matrix<double, 3, 1> &min_coords,
                               const Eigen::Matrix<double, 3, 1> &max_coords);
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