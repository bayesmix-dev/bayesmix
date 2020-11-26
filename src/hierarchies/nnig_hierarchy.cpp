#include "nnig_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/hierarchy_prior.pb.h"
#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"

void NNIGHierarchy::initialize() {
  assert(prior != nullptr && "Error: prior was not provided");
  state.mean = hypers->mean;
  state.var = hypers->scale / (hypers->shape + 1);
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
    post_params.mean = mu0;
    post_params.var_scaling = lambda0;
    post_params.shape = alpha0;
    post_params.scale = beta0;
    return post_params;
  }

  // Compute updated hyperparameters
  double y_bar = data.mean();  // sample mean
  post_params.mean = (lambda0 * mu0 + n * y_bar) / (lambda0 + n);
  post_params.var_scaling = lambda0 + n;
  post_params.shape = alpha0 + 0.5 * n;
  double ss = (data.dot(data)) - n * y_bar * y_bar;  // sum of squares
  post_params.scale =
      beta0 + 0.5 * ss +
      0.5 * lambda0 * n * (y_bar - mu0) * (y_bar - mu0) / (n + lambda0);

  return post_params;
}

void NNIGHierarchy::update_hypers(
    const std::vector<bayesmix::MarginalState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else if (prior->has_normal_mean_prior()) {
    // Get hyperparameters
    double mu00 = prior->normal_mean_prior().mean_prior().mean();
    double sig200 = prior->normal_mean_prior().mean_prior().var();
    double lambda0 = prior->normal_mean_prior().var_scaling();
    // Compute posterior hyperparameters
    double prec = 0.0;
    double num = 0.0;
    for (auto &st : states) {
      double mean = st.univ_ls_state().mean();
      double var = st.univ_ls_state().var();
      prec += 1 / var;
      num += mean / var;
    }
    prec = 1 / sig200 + lambda0 * prec;
    num = mu00 / sig200 + lambda0 * num;
    double mu_n = num / prec;
    double sig2_n = 1 / prec;
    // Update hyperparameters with posterior random sampling
    hypers->mean = stan::math::normal_rng(mu_n, sqrt(sig2_n), rng);
  }

  else if (prior->has_ngg_prior()) {
    // Get hyperparameters:
    // for mu0
    double mu00 = prior->ngg_prior().mean_prior().mean();
    double sig200 = prior->ngg_prior().mean_prior().var();
    // for lambda0
    double alpha00 = prior->ngg_prior().var_scaling_prior().shape();
    double beta00 = prior->ngg_prior().var_scaling_prior().rate();
    // for tau0
    double a00 = prior->ngg_prior().scale_prior().shape();
    double b00 = prior->ngg_prior().scale_prior().rate();
    // Compute posterior hyperparameters
    double b_n = 0.0;
    double num = 0.0;
    double beta_n = 0.0;
    for (auto &st : states) {
      double mean = st.univ_ls_state().mean();
      double var = st.univ_ls_state().var();
      b_n += 1 / var;
      num += mean / var;
      beta_n += (hypers->mean - mean) * (hypers->mean - mean) / var;
    }
    double var = hypers->var_scaling * b_n + 1 / sig200;
    b_n += b00;
    num = hypers->var_scaling * num + mu00 / sig200;
    beta_n = beta00 + 0.5 * beta_n;
    double sig_n = 1 / var;
    double mu_n = num / var;
    double alpha_n = alpha00 + 0.5 * states.size();
    double a_n = a00 + states.size() * hypers->shape;
    // Update hyperparameters with posterior random Gibbs sampling
    hypers->mean = stan::math::normal_rng(mu_n, sig_n, rng);
    hypers->var_scaling = stan::math::gamma_rng(alpha_n, beta_n, rng);
    hypers->scale = stan::math::gamma_rng(a_n, b_n, rng);
  }

  else {
    throw std::invalid_argument("Error: unrecognized prior");
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
  double sig_n = sqrt(hypers->scale * (hypers->var_scaling + 1) /
                      (hypers->shape * hypers->var_scaling));
  return stan::math::student_t_lpdf(datum(0), 2 * hypers->shape, hypers->mean,
                                    sig_n);
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
Eigen::VectorXd NNIGHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(hypers->scale * (hypers->var_scaling + 1) /
                      (hypers->shape * hypers->var_scaling));

  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute marginal for each data point
    result(i) = stan::math::student_t_lpdf(data(i, 0), 2 * hypers->shape,
                                           hypers->mean, sig_n);
  }
  return result;
}

void NNIGHierarchy::draw() {
  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(hypers->shape, hypers->scale, rng);
  state.mean = stan::math::normal_rng(
      hypers->mean, sqrt(state.var / hypers->var_scaling), rng);
}

//! \param data Column vector of data points
void NNIGHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Update values
  Hyperparams params =
      normal_invgamma_update(data.col(0), hypers->mean, hypers->shape,
                             hypers->scale, hypers->var_scaling);

  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  state.mean = stan::math::normal_rng(
      params.mean, sqrt(state.var / params.var_scaling), rng);
}

void NNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::MarginalState::ClusterState &>(state_);
  state.mean = statecast.univ_ls_state().mean();
  state.var = statecast.univ_ls_state().var();
}

void NNIGHierarchy::set_prior(const google::protobuf::Message &prior_) {
  auto &priorcast =
      google::protobuf::internal::down_cast<const bayesmix::NNIGPrior &>(
          prior_);
  prior = std::make_shared<bayesmix::NNIGPrior>(priorcast);
  hypers = std::make_shared<Hyperparams>();
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = prior->fixed_values().mean();
    hypers->var_scaling = prior->fixed_values().var_scaling();
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    assert(hypers->var_scaling > 0);
    assert(hypers->shape > 0);
    assert(hypers->scale > 0);
  }

  else if (prior->has_normal_mean_prior()) {
    // Set initial values
    hypers->mean = prior->normal_mean_prior().mean_prior().mean();
    hypers->var_scaling = prior->normal_mean_prior().var_scaling();
    hypers->shape = prior->normal_mean_prior().shape();
    hypers->scale = prior->normal_mean_prior().scale();
    // Check validity
    assert(hypers->var_scaling > 0);
    assert(hypers->shape > 0);
    assert(hypers->scale > 0);
  }

  else if (prior->has_ngg_prior()) {
    // Get hyperparameters:
    // for mu0
    double mu00 = prior->ngg_prior().mean_prior().mean();
    double sigma00 = prior->ngg_prior().mean_prior().var();
    // for lambda0
    double alpha00 = prior->ngg_prior().var_scaling_prior().shape();
    double beta00 = prior->ngg_prior().var_scaling_prior().rate();
    // for beta0
    double a00 = prior->ngg_prior().scale_prior().shape();
    double b00 = prior->ngg_prior().scale_prior().rate();
    // for alpha0
    double alpha0 = prior->ngg_prior().shape();
    // Check validity
    assert(sigma00 > 0);
    assert(alpha00 > 0);
    assert(beta00 > 0);
    assert(a00 > 0);
    assert(b00 > 0);
    assert(alpha0 > 0);
    // Set initial values
    hypers->mean = mu00;
    hypers->var_scaling = alpha00 / beta00;
    hypers->shape = alpha0;
    hypers->scale = a00 / b00;
  }

  else {
    throw std::invalid_argument("Error: unrecognized prior");
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

void NNIGHierarchy::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::NNIGPrior hypers_;
  hypers_.mutable_fixed_values()->set_mean(hypers->mean);
  hypers_.mutable_fixed_values()->set_var_scaling(hypers->var_scaling);
  hypers_.mutable_fixed_values()->set_shape(hypers->shape);
  hypers_.mutable_fixed_values()->set_scale(hypers->scale);

  google::protobuf::internal::down_cast<bayesmix::NNIGPrior *>(out)
      ->mutable_fixed_values()
      ->CopyFrom(hypers_.fixed_values());
}
