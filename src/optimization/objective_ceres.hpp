#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

// ObjectiveFunctorCeres class definition
template <int Dim> struct ObjectiveFunctorCeres {
  using PointType =
      typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type;
  using Vector = std::conditional_t<Dim == 2, Eigen::Vector2d, Eigen::Vector3d>;

  ObjectiveFunctorCeres(
      const std::array<int, Dim> &num_points, const Vector &min_coords,
      const Vector &max_coords,
      const std::vector<pcl::PointCloud<PointType>> &point_clouds,
      const int number_of_points, const bool both_directions,
      const double step_size);

  bool compute_residuals(const Eigen::VectorXd &x, Eigen::VectorXd &residuals) const;

  void df(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobian) const;

  int num_inputs() const;

  int num_outputs() const;

private:
  const std::array<int, Dim> num_map_points_;
  const Vector min_coords_;
  const Vector max_coords_;
  const std::vector<pcl::PointCloud<PointType>> point_clouds_;
  const int number_of_points_;
  const bool both_directions_;
  const double step_size_;
};

class ManualCostFunction : public ceres::CostFunction {
public:
  ManualCostFunction(ObjectiveFunctorCeres<2> *functor) : functor_(functor) {
    set_num_residuals(functor->num_outputs());
    mutable_parameter_block_sizes()->push_back(functor->num_inputs());
  }

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const override {
    Eigen::Map<const Eigen::VectorXd> params(parameters[0], functor_->num_inputs());

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

private:
  ObjectiveFunctorCeres<2> *functor_;
};
