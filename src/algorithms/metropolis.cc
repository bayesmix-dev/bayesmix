#include "metropolis.h"

#include <iostream>
#include <stan/math/prim/prob/multi_normal_lpdf.hpp>
#include <stan/math/prim/prob/multi_normal_rng.hpp>
#include <stan/math/prim/prob/uniform_rng.hpp>

#include "src/utils/rng.h"

Eigen::VectorXd Metropolis::standard_mean() const { return state; }

Eigen::VectorXd Metropolis::mala_mean() const {
  Eigen::VectorXd grad = (-1.0 / true_var) * state;
  for (int i = 0; i < data.rows(); i++) {
    grad += (data(i) - sigmoid(state.dot(covariates.row(i)))) *
            covariates.row(i);
  }
  return state + penal * grad;
}

double Metropolis::like_lpdf(const Eigen::VectorXd &alpha) {
  double lpdf = 0.0;
  for (int i = 0; i < data.size(); i++) {
    double sig = sigmoid(covariates.row(i).dot(alpha));
    lpdf += data(i) * std::log(sig) + (1.0 - data(i)) * std::log(1 - sig);
  }
  return lpdf;
}

double Metropolis::prior_lpdf(const Eigen::VectorXd &alpha) {
  auto mean = Eigen::VectorXd::Zero(dim);
  auto covar = true_var * Eigen::MatrixXd::Identity(dim, dim);
  return stan::math::multi_normal_lpdf(alpha, mean, covar);
}

Eigen::VectorXd Metropolis::draw_proposal() {
  auto &rng = bayesmix::Rng::Instance().get();
  auto covar = prop_var * Eigen::MatrixXd::Identity(dim, dim);
  return stan::math::multi_normal_rng(state, covar, rng);
}

void Metropolis::metropolis_hastings_step() {
  // Draw proposed state from proposal
  state_prop = draw_proposal();
  // Eigen::VectorXd state_prop = 
  double like_ratio = like_lpdf(state_prop) - like_lpdf(state);
  double prior_ratio = prior_lpdf(state_prop) - prior_lpdf(state);
  // double proposal_ratio = h(alpha) - h(alpha*)
  logratio = like_ratio + prior_ratio;
  // Accept with probability ratio
  double p = stan::math::uniform_rng(0.0, 1.0, rng);
  if (p <= std::exp(logratio)) {
    state = state_prop;
  }
  output();
}

void Metropolis::output() {
  std::cout << "#" << iter << ":\tp=" << ratio
            << ",\tstate=" << state.transpose()
            << std::endl;
}
