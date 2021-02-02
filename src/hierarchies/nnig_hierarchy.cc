#include "nnig_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "marginal_state.pb.h"
#include "src/utils/rng.h"

void NNIGHierarchy::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Hierarchy prior was not provided");
  }
  state.mean = hypers->mean;
  state.var = hypers->scale / (hypers->shape + 1);
}

//! \param data                        Column vector of data points
//! \param mu0, alpha0, beta0, lambda0 Original values for hyperparameters
//! \return                            Vector of updated values for hyperpar.s
NNIGHierarchy::Hyperparams NNIGHierarchy::normal_invgamma_update() {
  Hyperparams post_params;
  if (card == 0) {  // no update possible
    post_params = *hypers;
    return post_params;
  }
  // Compute updated hyperparameters
  double y_bar = data_sum / (1.0 * card);  // sample mean
  double ss = data_sum_squares - card * y_bar * y_bar;
  post_params.mean = (hypers->var_scaling * hypers->mean + data_sum) /
                     (hypers->var_scaling + card);
  post_params.var_scaling = hypers->var_scaling + card;
  post_params.shape = hypers->shape + 0.5 * card;
  post_params.scale = hypers->scale + 0.5 * ss +
                      0.5 * hypers->var_scaling * card *
                          (y_bar - hypers->mean) * (y_bar - hypers->mean) /
                          (card + hypers->var_scaling);
  return post_params;
}

void NNIGHierarchy::clear_data() {
  data_sum = 0;
  data_sum_squares = 0;
  card = 0;
  cluster_data_idx = std::set<int>();
}

void NNIGHierarchy::update_summary_statistics(const Eigen::VectorXd &datum,
                                              bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
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
      double mean = st.uni_ls_state().mean();
      double var = st.uni_ls_state().var();
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
      double mean = st.uni_ls_state().mean();
      double var = st.uni_ls_state().var();
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
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

//! \param data Column vector containing a single data point
//! \return     Log-Likehood vector evaluated in data
double NNIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum,
                                const Eigen::RowVectorXd &covariate) const {
  return stan::math::normal_lpdf(datum(0), state.mean, sqrt(state.var));
}

void NNIGHierarchy::draw() {
  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(hypers->shape, hypers->scale, rng);
  state.mean = stan::math::normal_rng(
      hypers->mean, sqrt(state.var / hypers->var_scaling), rng);
}

//! \param data Column vector of data points
void NNIGHierarchy::sample_given_data() {
  Hyperparams params = normal_invgamma_update();

  // Update state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  state.mean = stan::math::normal_rng(
      params.mean, sqrt(state.var / params.var_scaling), rng);
}

void NNIGHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  data_sum = data.sum();
  data_sum_squares = data.squaredNorm();
  card = data.rows();
  log_card = std::log(card);
  sample_given_data();
}

void NNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::MarginalState::ClusterState &>(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.var = statecast.uni_ls_state().var();
  set_card(statecast.cardinality());
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
    if (hypers->var_scaling <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
  }

  else if (prior->has_normal_mean_prior()) {
    // Set initial values
    hypers->mean = prior->normal_mean_prior().mean_prior().mean();
    hypers->var_scaling = prior->normal_mean_prior().var_scaling();
    hypers->shape = prior->normal_mean_prior().shape();
    hypers->scale = prior->normal_mean_prior().scale();
    // Check validity
    if (hypers->var_scaling <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
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
    if (sigma00 <= 0) {
      throw std::invalid_argument("Variance parameter must be > 0");
    }
    if (alpha00 <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (beta00 <= 0) {
      throw std::invalid_argument("Rate parameter must be > 0");
    }
    if (a00 <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (b00 <= 0) {
      throw std::invalid_argument("Rate parameter must be > 0");
    }
    if (alpha0 <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    // Set initial values
    hypers->mean = mu00;
    hypers->var_scaling = alpha00 / beta00;
    hypers->shape = alpha0;
    hypers->scale = a00 / b00;
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void NNIGHierarchy::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::UniLSState state_;
  state_.set_mean(state.mean);
  state_.set_var(state.var);

  auto *out_cast = google::protobuf::internal::down_cast<
      bayesmix::MarginalState::ClusterState *>(out);
  out_cast->mutable_uni_ls_state()->CopyFrom(state_);
  out_cast->set_cardinality(card);
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
