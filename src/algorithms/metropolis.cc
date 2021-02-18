#include "metropolis.h"

#include <iostream>
#include <stan/math/prim/prob/bernoulli_rng.hpp>
#include <stan/math/prim/prob/multi_normal_lpdf.hpp>
#include <stan/math/prim/prob/multi_normal_rng.hpp>
#include <stan/math/prim/prob/uniform_rng.hpp>

#include "src/utils/rng.h"

void Metropolis::generate_data() {
  int n_data = 500;
  // Initialize true coefficients
  dim = 3;
  Eigen::VectorXd alpha_true(dim);
  alpha_true << -5.0, 5.0, 0.0;
  // Generate data
  covariates = Eigen::MatrixXd::Random(n_data, dim);
  Eigen::VectorXd y(n_data);
  auto &rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < n_data; i++) {
    double prob = sigmoid(covariates.row(i) * alpha_true);
    y(i) = stan::math::bernoulli_rng(prob, rng);
  }
  data = y;
}

Eigen::VectorXd Metropolis::gradient(const Eigen::VectorXd &alpha) const {
  Eigen::VectorXd grad = (-1.0 / true_var) * alpha;
  for (int i = 0; i < data.rows(); i++) {
    grad +=
        (data(i) - sigmoid(alpha.dot(covariates.row(i)))) * covariates.row(i);
  }
  return grad;
}

Eigen::VectorXd Metropolis::draw_proposal() const {
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd mean;
  Eigen::MatrixXd covar;
  if (use_mala) {
    mean = state + step * gradient(state);
    covar = std::sqrt(2.0 * step) * Eigen::MatrixXd::Identity(dim, dim);
  } else {
    mean = state;
    covar = prop_var * Eigen::MatrixXd::Identity(dim, dim);
  }
  return stan::math::multi_normal_rng(mean, covar, rng);
}

double Metropolis::like_lpdf(const Eigen::VectorXd &alpha) const {
  double lpdf = 0.0;
  for (int i = 0; i < data.size(); i++) {
    double sig = sigmoid(covariates.row(i).dot(alpha));
    lpdf += data(i) * std::log(sig) + (1.0 - data(i)) * std::log(1 - sig);
  }
  return lpdf;
}

double Metropolis::prior_lpdf(const Eigen::VectorXd &alpha) const {
  auto mean = Eigen::VectorXd::Zero(dim);
  auto covar = true_var * Eigen::MatrixXd::Identity(dim, dim);
  return stan::math::multi_normal_lpdf(alpha, mean, covar);
}

double Metropolis::proposal_lpdf(const Eigen::VectorXd &alpha) const {
  Eigen::VectorXd mean = alpha + step * gradient(alpha);
  auto covar = std::sqrt(2.0 * step) * Eigen::MatrixXd::Identity(dim, dim);
  return stan::math::multi_normal_lpdf(alpha, mean, covar);
}

void Metropolis::metropolis_hastings_step() {
  auto &rng = bayesmix::Rng::Instance().get();
  // Create proposed state
  Eigen::VectorXd state_prop = draw_proposal();
  // Compute acceptance ratio
  double like_ratio = like_lpdf(state_prop) - like_lpdf(state);
  double prior_ratio = prior_lpdf(state_prop) - prior_lpdf(state);
  double proposal_ratio = 0.0;
  if (use_mala) {
    proposal_ratio = proposal_lpdf(state_prop) - proposal_lpdf(state);
  }
  logratio = like_ratio + prior_ratio - proposal_ratio;
  // Accept with probability ratio
  double p = stan::math::uniform_rng(0.0, 1.0, rng);
  if (p < std::exp(logratio)) {
    accepted = true;
    state = state_prop;
  } else {
    accepted = false;
  }
  output();
}

void Metropolis::output() {
  std::cout << "#" << iter << ":\tp=" << std::exp(logratio);
  if (accepted) {
    std::cout << ",\tstate=" << state.transpose();
  }
  std::cout << std::endl;
}
