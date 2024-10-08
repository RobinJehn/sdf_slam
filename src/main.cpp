#include "optimization/objective.hpp"
#include <Eigen/Dense>
#include <iostream>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

int main(int argc, char *argv[]) {
  Eigen::VectorXd x(2);
  x(0) = -10.0;
  x(1) = 3.0;
  std::cout << "x: " << x << std::endl;

  // ObjectiveFunctor functor();
  // Eigen::NumericalDiff<ObjectiveFunctor> numDiff(functor);
  // Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ObjectiveFunctor>, double> lm(
  //     numDiff);
  // lm.parameters.maxfev = 2000;
  // lm.parameters.xtol = 1.0e-10;
  // std::cout << lm.parameters.maxfev << std::endl;

  // int ret = lm.minimize(x);
  // std::cout << lm.iter << std::endl;
  // std::cout << ret << std::endl;

  // std::cout << "x that minimizes the function: " << x << std::endl;

  // Eigen::VectorXd y(2);
  // functor(x, y);

  // std::cout << "f(x) = " << y << std::endl;

  // std::cout << "press [ENTER] to continue " << std::endl;
  // std::cin.get();
  return 0;
}