#include "nnig_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/hierarchy_prior.pb.h"
#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"

void NNIGHierarchy::initialize() {
  state.mean = hypers->mu;
  state.var = hypers->beta / (hypers->alpha + 1);
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
    const std::vector<bayesmix::MarginalState::ClusterState> &states) {
  if (prior.has_fixed_values()) {
    return;
  } else if (prior.has_normal_mean_prior()) {
    // Get hyperparameters
    double mu00 = prior.normal_mean_prior().mean_prior().mean();
    double sig200 = prior.normal_mean_prior().mean_prior().var();
    double lambda0 = prior.normal_mean_prior().var_scaling();

    // Compute posterior hyperparameters
    double prec = 0.0;
    double num = 0.0;
    for (auto &st : states) {
      double mean = st.univ_ls_state().mean();
      double var = st.univ_ls_state().var();
      double prec_i = 1 / var;
      prec += prec_i;
      num += mean * prec_i;
    }
    prec = 1 / sig200 + lambda0 * prec;
    num = mu00 / sig200 + lambda0 * num;

    double mu_n = num / prec;
    double sig2_n = 1 / prec;

    // Update hyperparameters with posterior random sampling
    auto &rng = bayesmix::Rng::Instance().get();
    hypers->mu = stan::math::normal_rng(mu_n, sqrt(sig2_n), rng);

  } else {
    std::invalid_argument("Error: unrecognized prior");
  }
}

//! \param data Column vector containing a single data point
//! \return     Log-Likehood vector evaluated in data
double NNIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::normal_lpdf(datum(0), state.mean, sqrt(state.var));
}

//! \param data Column vector of data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd NNIGHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute likelihood for each data point
    result(i) =
        stan::math::normal_lpdf(data(i, 0), state.mean, sqrt(state.var));
  }
  return result;
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
double NNIGHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) const {
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
  state.var = stan::math::inv_gamma_rng(hypers->alpha, hypers->beta, rng);
  state.mean = stan::math::normal_rng(hypers->mu,
                                      sqrt(state.var / hypers->lambda), rng);
}

//! \param data Column vector of data points
void NNIGHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Update values
  Hyperparams params = normal_invgamma_update(
      data.col(0), hypers->mu, hypers->alpha, hypers->beta, hypers->lambda);

  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(params.alpha, params.beta, rng);
  state.mean =
      stan::math::normal_rng(params.mu, sqrt(state.var / params.lambda), rng);
}

void NNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  const bayesmix::MarginalState::ClusterState &currcast =
      google::protobuf::internal::down_cast<
          const bayesmix::MarginalState::ClusterState &>(state_);

  state.mean = currcast.univ_ls_state().mean();
  state.var = currcast.univ_ls_state().var();
}

void NNIGHierarchy::set_prior(const google::protobuf::Message &prior_) {
  const bayesmix::NNIGPrior &currcast =
      google::protobuf::internal::down_cast<const bayesmix::NNIGPrior &>(
          prior_);
  prior = currcast;
  hypers = std::make_shared<Hyperparams>();
  if (prior.has_fixed_values()) {
    // Check validity
    assert(prior.fixed_values().var_scaling() > 0);
    assert(prior.fixed_values().shape() > 0);
    assert(prior.fixed_values().rate() > 0);
    // Set values
    hypers->mu = prior.fixed_values().mean();
    hypers->lambda = prior.fixed_values().var_scaling();
    hypers->alpha = prior.fixed_values().shape();
    hypers->beta = prior.fixed_values().rate();
  } else if (prior.has_normal_mean_prior()) {
    // Check validity
    assert(prior.normal_mean_prior().var_scaling() > 0);
    assert(prior.normal_mean_prior().shape() > 0);
    assert(prior.normal_mean_prior().rate() > 0);
    // Set initial values
    hypers->mu = prior.normal_mean_prior().mean_prior().mean();
    hypers->lambda = prior.normal_mean_prior().var_scaling();
    hypers->alpha = prior.normal_mean_prior().shape();
    hypers->beta = prior.normal_mean_prior().rate();
  } else {
    std::invalid_argument("Error: unrecognized prior");
  }
}

void NNIGHierarchy::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::UnivLSState state_;
  state_.set_mean(state.mean);
  state_.set_var(state.var);

  google::protobuf::internal::down_cast<
      bayesmix::MarginalState::ClusterState *>(out)
      ->mutable_univ_ls_state()
      ->CopyFrom(state_);
}
