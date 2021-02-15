#include "metropolis.h"

#include <stan/math/prim.hpp>

double Metropolis::standard_mean() const {
  return state;
}

double Metropolis::mala_mean() const {
  Eigen::VectorXd grad = (-1.0 / true_var) * alpha;
  for (int i = 0; i < data.rows(); i++) {
    grad +=
      (data(i) - inv_logit(state.dot(covariates.row(i)))) * covariates.row(i);
  }
  return state + tau * grad;
}

void Metropolis::metropolis_hastings_step() {
  // Draw proposed state from proposal
  Eigen::VectorXd mean = use_mala : mala_mean() ? standard_mean();
  auto covar = var * Eigen::MatrixXd::Identity(dim, dim);
  Eigen::VectorXd proposed = stan::math::multi_normal_rng(mean, covar, rng);
  double ratio1 = std::exp((-0.5/true_var)*(proposed.dot(proposed) -
    state.dot(state)));
  double ratio2 = 1.0;
  for (int i = 0; i < data.rows(); i++) {
    ratio2 *= std::exp(data.row(i) * covariates.row(i).dot(proposed)) /
      (1+std::exp(covariates.row(i).dot(proposed)));
  }
  double ratio3 = std::exp(
    (-0.5 / prop_var) *
      ( (mean-proposed).dot(mean-proposed) - (mean-alpha).dot(mean-alpha) )
    );
  double ratio = ratio1 * ratio2 * ratio3;
  // Accept with probability ratio
  double p = stan::math::uniform_rng(0.0, 1.0, rng);
  if (p <= ratio) {
    state = proposed;
  }
}

void Metropolis::output() {
  std::cout << "#" << iter << ":\t";
  for (int i = 0; i < dim; i++) {
    std::cout << state(i) << " ";
  }
  std::cout << std::endl;
}
