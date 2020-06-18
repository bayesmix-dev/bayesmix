#ifndef HIERARCHYNNIG_IMP_HPP
#define HIERARCHYNNIG_IMP_HPP

#include "HierarchyNNIG.hpp"

template <class Hypers>
void HierarchyNNIG<Hypers>::check_state_validity() {
  assert(state[1](0, 0) > 0);
}

//! \param data                       Column vector of data points
//! \param mu0, alpha0, beta0, lambda Original values for hyperparameters
//! \return                           Vector of updated values for hyperpar.s
template <class Hypers>
std::vector<double> HierarchyNNIG<Hypers>::normal_gamma_update(
    const Eigen::VectorXd &data, const double mu0, const double alpha0,
    const double beta0, const double lambda) {
  // Initialize relevant variables
  double mu_post, alpha_post, beta_post, lambda_post;
  unsigned int n = data.rows();

  if (n == 0) {  // no update possible
    return std::vector<double>{mu0, alpha0, beta0, lambda};
  }

  // Compute updated hyperparameters
  double y_bar = data.mean();  // sample mean
  mu_post = (lambda * mu0 + n * y_bar) / (lambda + n);
  alpha_post = alpha0 + 0.5 * n;
  double ss = (data.dot(data)) - n * y_bar * y_bar;  // sum of squares
  beta_post = beta0 + 0.5 * ss +
              0.5 * lambda * n * (y_bar - mu0) * (y_bar - mu0) / (n + lambda);
  lambda_post = lambda + n;

  return std::vector<double>{mu_post, alpha_post, beta_post, lambda_post};
}

//! \param data Column vector of data points
//! \return     Likehood vector evaluated in data
template <class Hypers>
Eigen::VectorXd HierarchyNNIG<Hypers>::like(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = lpdf(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Log-Likehood vector evaluated in data
template <class Hypers>
Eigen::VectorXd HierarchyNNIG<Hypers>::lpdf(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute likelihood for each data point
    result(i) =
        stan::math::normal_lpdf(data(i, 0), state[0](0, 0), state[1](0, 0));
  }
  return result;
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data
template <class Hypers>
Eigen::VectorXd HierarchyNNIG<Hypers>::eval_marg(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = marg_lpdf(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
template <class Hypers>
Eigen::VectorXd HierarchyNNIG<Hypers>::marg_lpdf(const Eigen::MatrixXd &data) {
  // Get values of hyperparameters
  double mu0 = hypers->get_mu0();
  double lambda = hypers->get_lambda();
  double alpha0 = hypers->get_alpha0();
  double beta0 = hypers->get_beta0();

  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(beta0 * (lambda + 1) / (alpha0 * lambda));

  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute marginal for each data point
    result(i) = stan::math::student_t_lpdf(data(i, 0), 2 * alpha0, mu0, sig_n);
  }
  return result;
}

template <class Hypers>
void HierarchyNNIG<Hypers>::draw() {
  // Get values of hyperparameters
  double mu0 = hypers->get_mu0();
  double lambda = hypers->get_lambda();
  double alpha0 = hypers->get_alpha0();
  double beta0 = hypers->get_beta0();

  // Generate new state values from their prior centering distribution
  double sig_new =
      sqrt(stan::math::inv_gamma_rng(alpha0, beta0, Rng::Instance().get()));
  double mu_new = stan::math::normal_rng(mu0, sig_new / sqrt(lambda),
                                         Rng::Instance().get());

  // Update state
  state[0](0, 0) = mu_new;
  state[1](0, 0) = sig_new;
}

//! \param data Column vector of data points
template <class Hypers>
void HierarchyNNIG<Hypers>::sample_given_data(const Eigen::MatrixXd &data) {
  // Get values of hyperparameters
  double mu0 = hypers->get_mu0();
  double lambda = hypers->get_lambda();
  double alpha0 = hypers->get_alpha0();
  double beta0 = hypers->get_beta0();

  // Update values
  std::vector<double> temp =
      normal_gamma_update(data.col(0), mu0, alpha0, beta0, lambda);
  double mu_post = temp[0];
  double alpha_post = temp[1];
  double beta_post = temp[2];
  double lambda_post = temp[3];

  // Generate new state values from their prior centering distribution
  double sig_new = sqrt(
      stan::math::inv_gamma_rng(alpha_post, beta_post, Rng::Instance().get()));
  double mu_new = stan::math::normal_rng(mu_post, sig_new / sqrt(lambda_post),
                                         Rng::Instance().get());

  // Update state
  state[0](0, 0) = mu_new;
  state[1](0, 0) = sig_new;
}

#endif  // HIERARCHYNNIG_IMP_HPP
