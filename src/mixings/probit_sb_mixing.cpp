#include "probit_sb_mixing.hpp"
#include <Eigen/Dense>

ProbitSBMixing::initialize() {
  state.coefficients = Eigen::VectorXd::Zero(dim);
}
