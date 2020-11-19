#include "nnig_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"

void NNIGHierarchy::check_and_initialize() {
  check_hypers_validity();
  mean = hypers->mu;
  sd = sqrt(hypers->beta / (hypers->alpha - 1));
}

//! \param data                        Column vector of data points
//! \param mu0, alpha0, beta0, lambda0 Original values for hyperparameters
//! \return                            Vector of updated values for hyperpar.s
NNIGHierarchy::Hyperparams NNIGHierarchy::normal_invgamma_update(
    const Eigen::VectorXd &data, const double mu0, const double alpha0,
    const double beta0, const double lambda0) {
  // Initialize relevant variables
  Hyperparams post_params;

  unsigned int n = data.rows();

  if (n == 0) {  // no update possible
    post_params.mu = mu0;
    post_params.alpha = alpha0;
    post_params.beta = beta0;
    post_params.lambda = lambda0;
    return post_params;
  }

  // Compute updated hyperparameters
  double y_bar = data.mean();  // sample mean
  post_params.mu = (lambda0 * mu0 + n * y_bar) / (lambda0 + n);
  post_params.alpha = alpha0 + 0.5 * n;
  double ss = (data.dot(data)) - n * y_bar * y_bar;  // sum of squares
  post_params.beta =
      beta0 + 0.5 * ss +
      0.5 * lambda0 * n * (y_bar - mu0) * (y_bar - mu0) / (n + lambda0);
  post_params.lambda = lambda0 + n;

  return post_params;
}

void NNIGHierarchy::update_hypers(
    const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
    unsigned int n) {
  return;
}

//! \param data Column vector containing a single data point
//! \return     Log-Likehood vector evaluated in data
double NNIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  assert(datum.size() == 1);
  return stan::math::normal_lpdf(datum(0), mean, sd);
}

//! \param data Column vector of data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd NNIGHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute likelihood for each data point
    result(i) = stan::math::normal_lpdf(data(i, 0), mean, sd);
  }
  return result;
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
double NNIGHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) const {
  assert(datum.size() == 1);

  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(hypers->beta * (hypers->lambda + 1) /
                      (hypers->alpha * hypers->lambda));
  return stan::math::student_t_lpdf(datum(0), 2 * hypers->alpha, hypers->mu,
                                    sig_n);
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
Eigen::VectorXd NNIGHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(hypers->beta * (hypers->lambda + 1) /
                      (hypers->alpha * hypers->lambda));

  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute marginal for each data point
    result(i) = stan::math::student_t_lpdf(data(i, 0), 2 * hypers->alpha,
                                           hypers->mu, sig_n);
  }
  return result;
}

void NNIGHierarchy::draw() {
  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  sd = sqrt(stan::math::inv_gamma_rng(hypers->alpha, hypers->beta, rng));
  mean = stan::math::normal_rng(hypers->mu, sd / sqrt(hypers->lambda), rng);
}

//! \param data Column vector of data points
void NNIGHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Update values
  Hyperparams params = normal_invgamma_update(
      data.col(0), hypers->mu, hypers->alpha, hypers->beta, hypers->lambda);

  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  sd = sqrt(stan::math::inv_gamma_rng(params.alpha, params.beta, rng));
  mean = stan::math::normal_rng(params.mu, sd / sqrt(params.lambda), rng);
}

void NNIGHierarchy::set_state(const google::protobuf::Message &state_,
                              bool check /*= true*/) {
  const bayesmix::MarginalState::ClusterVal &currcast =
      google::protobuf::internal::down_cast<
          const bayesmix::MarginalState::ClusterVal &>(state_);

  mean = currcast.univ_ls_state().mean();
  sd = currcast.univ_ls_state().sd();

  if (check) {
    check_state_validity();
  }
}

void NNIGHierarchy::set_prior(const google::protobuf::Message &prior_) {
  return;
}

void NNIGHierarchy::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::UnivLSState state;
  state.set_mean(mean);
  state.set_sd(sd);

  google::protobuf::internal::down_cast<bayesmix::MarginalState::ClusterVal *>(
      out)
      ->mutable_univ_ls_state()
      ->CopyFrom(state);
}
