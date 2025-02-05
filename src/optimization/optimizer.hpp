#pragma once

#include <Eigen/Core>
#include <functional>  // For std::function
#include <unsupported/Eigen/NonLinearOptimization>

template <typename FunctorType, typename Scalar = double>
class LevenbergMarquardtWithCallback {
 public:
  using FVectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using CallbackType = std::function<void(const FVectorType &, int, Scalar)>;

  LevenbergMarquardtWithCallback(FunctorType &functor, CallbackType callback);

  // Function to run the minimization and invoke the callback after each step
  Eigen::LevenbergMarquardtSpace::Status minimize(FVectorType &x);

  Eigen::LevenbergMarquardt<FunctorType, Scalar> lm;

 private:
  CallbackType callback;
};
