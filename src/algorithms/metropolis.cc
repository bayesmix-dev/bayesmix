#include "metropolis.h"

#include <stan/math/prim.hpp>

// MALA: invece che in alpha, la proposal viene centrata in
// alpha + tau * grad(log(target(alpha)))

double Metropolis::standard_mean() const {
  return state;
}

double Metropolis::mala_mean() const {
  return state;  // TODO
}

void Metropolis::metropolis_hastings_step() {
  // Draw proposed state from proposal
  Eigen::VectorXd mean = use_mala : mala_mean() ? standard_mean();
  auto covar = var * Eigen::MatrixXd::Identity(dim, dim);
  Eigen::VectorXd proposed = stan::math::multi_normal_rng(mean, covar, rng);
  // Compute acceptance ratio
  double ratio = 1.0;  // TODO remember min!
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
