#include "nnig_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"

void NNIGHierarchy::check_and_initialize() {
  check_hypers_validity();
  mean = get_mu0();
  sd = sqrt(get_beta0() / (get_alpha0() - 1));
}

//! \param data                        Column vector of data points
//! \param mu0, alpha0, beta0, lambda0 Original values for hyperparameters
//! \return                            Vector of updated values for hyperpar.s
std::vector<double> NNIGHierarchy::normal_gamma_update(
    const Eigen::VectorXd &data, const double mu0, const double alpha0,
    const double beta0, const double lambda0) {
  // Initialize relevant variables
  double mu_post, alpha_post, beta_post, lambda_post;
  unsigned int n = data.rows();

  if (n == 0) {  // no update possible
    return std::vector<double>{mu0, alpha0, beta0, lambda0};
  }

  // Compute updated hyperparameters
  double y_bar = data.mean();  // sample mean
  mu_post = (lambda0 * mu0 + n * y_bar) / (lambda0 + n);
  alpha_post = alpha0 + 0.5 * n;
  double ss = (data.dot(data)) - n * y_bar * y_bar;  // sum of squares
  beta_post = beta0 + 0.5 * ss +
              0.5 * lambda0 * n * (y_bar - mu0) * (y_bar - mu0) / (n + lambda0);
  lambda_post = lambda0 + n;

  return std::vector<double>{mu_post, alpha_post, beta_post, lambda_post};
}

//! \param data Column vector of data points
//! \return     Likehood vector evaluated in data
Eigen::VectorXd NNIGHierarchy::like(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = lpdf(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd NNIGHierarchy::lpdf(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute likelihood for each data point
    result(i) = stan::math::normal_lpdf(data(i, 0), mean, sd);
  }
  return result;
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data
Eigen::VectorXd NNIGHierarchy::eval_marg(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = marg_lpdf(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
Eigen::VectorXd NNIGHierarchy::marg_lpdf(const Eigen::MatrixXd &data) {
  // Get values of hyperparameters
  double mu0 = get_mu0();
  double lambda0 = get_lambda0();
  double alpha0 = get_alpha0();
  double beta0 = get_beta0();

  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(beta0 * (lambda0 + 1) / (alpha0 * lambda0));

  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute marginal for each data point
    result(i) = stan::math::student_t_lpdf(data(i, 0), 2 * alpha0, mu0, sig_n);
  }
  return result;
}

void NNIGHierarchy::draw() {
  // Get values of hyperparameters
  double mu0 = get_mu0();
  double lambda0 = get_lambda0();
  double alpha0 = get_alpha0();
  double beta0 = get_beta0();

  // Update state values from their prior centering distribution
  auto rng = bayesmix::Rng::Instance().get();
  sd = sqrt(stan::math::inv_gamma_rng(alpha0, beta0, rng));
  mean = stan::math::normal_rng(mu0, sd / sqrt(lambda0), rng);
}

//! \param data Column vector of data points
void NNIGHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Update values
  std::vector<double> temp = normal_gamma_update(
      data.col(0), get_mu0(), get_alpha0(), get_beta0(), get_lambda0());
  double mu_post = temp[0];
  double alpha_post = temp[1];
  double beta_post = temp[2];
  double lambda_post = temp[3];

  // Update state values from their prior centering distribution
  auto rng = bayesmix::Rng::Instance().get();
  sd = sqrt(stan::math::inv_gamma_rng(alpha_post, beta_post, rng));
  mean = stan::math::normal_rng(mu_post, sd / sqrt(lambda_post), rng);
}

void NNIGHierarchy::set_state(google::protobuf::Message *curr, bool check) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  MarginalState::ClusterVal *currcast =
      down_cast<MarginalState::ClusterVal *>(curr);

  mean = currcast->univ_ls_state().mean();
  sd = currcast->univ_ls_state().sd();

  if (check) {
    check_state_validity();
  }
}

void NNIGHierarchy::write_state_to_proto(google::protobuf::Message *out) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  UnivLSState state;
  state.set_mean(mean);
  state.set_sd(sd);

  down_cast<MarginalState::ClusterVal *>(out)
      ->mutable_univ_ls_state()
      ->CopyFrom(state);
}
