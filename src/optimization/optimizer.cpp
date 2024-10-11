#include "optimizer.hpp"
#include "objective.hpp"

template <typename FunctorType, typename Scalar>
LevenbergMarquardtWithCallback<
    FunctorType, Scalar>::LevenbergMarquardtWithCallback(FunctorType &functor,
                                                         CallbackType callback)
    : lm(functor), callback(callback) {}

template <typename FunctorType, typename Scalar>
Eigen::LevenbergMarquardtSpace::Status
LevenbergMarquardtWithCallback<FunctorType, Scalar>::minimize(FVectorType &x) {
  Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(x);

  // Run the optimization in steps
  if (status == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
    return status;
  do {
    if (callback) {
      callback(x, lm.iter, lm.fnorm);
    }

    status = lm.minimizeOneStep(x);
  } while (status == Eigen::LevenbergMarquardtSpace::Running);

  return status;
}

// Explicit template instantiation for specific types
template class LevenbergMarquardtWithCallback<
    Eigen::NumericalDiff<ObjectiveFunctor<2>>, double>;
template class LevenbergMarquardtWithCallback<ObjectiveFunctor<2>, double>;
