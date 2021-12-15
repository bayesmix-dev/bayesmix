#include "lapnig_hierarchy.h"
#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/rng.h"

void LapNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.scale = statecast.uni_ls_state().scale();
  set_card(statecast.cardinality());
}

void LapNIGHierarchy::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).nnig_state();
  hypers->mean = hyperscast.mean();
  hypers->var_scaling = hyperscast.var_scaling();
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
}

double LapNIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::double_exponential_lpdf(datum(0), state.mean,
                                             state.scale);
}


void LapNIGHierarchy::update_summary_statistics(const Eigen::RowVectorXd &datum,
                                                bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
LapNIGHierarchy::get_state_proto() const {
  bayesmix::UniLSState state_;
  state_.set_mean(state.mean);
  state_.set_var(state.scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_uni_ls_state()->CopyFrom(state_);
  return out;
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
LapNIGHierarchy::get_hypers_proto() const {
  bayesmix::NNIGState hypers_;
  hypers_.set_mean(hypers->mean);
  hypers_.set_var_scaling(hypers->var_scaling);
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_nnig_state()->CopyFrom(hypers_);
  return out;
}

void LapNIGHierarchy::clear_summary_statistics() {
  data_sum = 0;
  data_sum_squares = 0;
}

void LapNIGHierarchy::initialize_hypers() {
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

void LapNIGHierarchy::initialize_state() {
  state.mean = hypers->mean;
  state.scale = hypers->scale / (hypers->shape + 1);
}

void LapNIGHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
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

LapNIG::State LapNIGHierarchy::draw(const LapNIG::Hyperparams &params) {
  auto &rng = bayesmix::Rng::Instance().get();
  LapNIG::State out;
  out.scale = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  out.mean = stan::math::normal_rng(params.mean,
                                    sqrt(params.var_scaling), rng);
  return out;
}
