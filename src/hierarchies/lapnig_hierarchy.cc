#include "lapnig_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/rng.h"

unsigned int LapNIGHierarchy::accepted_ = 0;
unsigned int LapNIGHierarchy::iter_ = 0;

void LapNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.scale = statecast.uni_ls_state().var();
  set_card(statecast.cardinality());
}

void LapNIGHierarchy::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).lapnig_state();
  hypers->mean = hyperscast.mean();
  hypers->var = hyperscast.var();
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
  hypers->mh_mean_var = hyperscast.mh_mean_var();
  hypers->mh_log_scale_var = hyperscast.mh_log_scale_var();
}

double LapNIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::double_exponential_lpdf(datum(0), state.mean,
                                             state.scale);
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
  bayesmix::LapNIGState hypers_;
  hypers_.set_mean(hypers->mean);
  hypers_.set_var(hypers->var);
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);
  hypers_.set_mh_mean_var(hypers->mh_mean_var);
  hypers_.set_mh_log_scale_var(hypers->mh_log_scale_var);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_lapnig_state()->CopyFrom(hypers_);
  return out;
}

void LapNIGHierarchy::clear_summary_statistics() {
  cluster_data_values.clear();
  sum_abs_diff_curr = 0;
  sum_abs_diff_prop = 0;
}

void LapNIGHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = prior->fixed_values().mean();
    hypers->var = prior->fixed_values().var();
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    hypers->mh_mean_var = prior->fixed_values().mh_mean_var();
    hypers->mh_log_scale_var = prior->fixed_values().mh_log_scale_var();
    // Check validity
    if (hypers->var <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
    if (hypers->mh_mean_var <= 0) {
      throw std::invalid_argument("mh_mean_var parameter must be > 0");
    }
    if (hypers->mh_log_scale_var <= 0) {
      throw std::invalid_argument("mh_log_scale_var parameter must be > 0");
    }
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void LapNIGHierarchy::initialize_state() {
  state.mean = hypers->mean;
  state.scale = hypers->scale / (hypers->shape + 1);  // mode of Inv-Gamma
}

void LapNIGHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

LapNIG::State LapNIGHierarchy::draw(const LapNIG::Hyperparams &params) {
  auto &rng = bayesmix::Rng::Instance().get();
  LapNIG::State out{};
  out.scale = stan::math::inv_gamma_rng(params.shape, 1. / params.scale, rng);
  out.mean = stan::math::normal_rng(params.mean, sqrt(params.var), rng);
  return out;
}

void LapNIGHierarchy::update_summary_statistics(
    const Eigen::RowVectorXd &datum, bool add) {
  if (add) {
    sum_abs_diff_curr += std::abs(state.mean - datum(0, 0));
    cluster_data_values.push_back(datum);
  } else {
    sum_abs_diff_curr -= std::abs(state.mean - datum(0, 0));
    auto it = std::find(cluster_data_values.begin(), cluster_data_values.end(),
                        datum);
    cluster_data_values.erase(it);
  }
}

void LapNIGHierarchy::sample_full_cond(bool update_params) {
  if (this->card == 0) {
    // No posterior update possible
    this->sample_prior();
  } else {
    // Number of iterations to compute the acceptance rate
    ++iter_;

    // Random generator
    auto &rng = bayesmix::Rng::Instance().get();

    // Candidate mean and candidate log_scale
    Eigen::VectorXd curr_unc_params(2);
    curr_unc_params << state.mean, std::log(state.scale);

    Eigen::VectorXd prop_unc_params = propose_rwmh(curr_unc_params);

    double log_target_prop =
        eval_prior_lpdf_unconstrained(prop_unc_params) +
        eval_like_lpdf_unconstrained(prop_unc_params, false);

    double log_target_curr =
        eval_prior_lpdf_unconstrained(curr_unc_params) +
        eval_like_lpdf_unconstrained(curr_unc_params, true);

    double log_a_rate = log_target_prop - log_target_curr;

    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_a_rate) {
      ++accepted_;
      state.mean = prop_unc_params(0);
      state.scale = std::exp(prop_unc_params(1));
      sum_abs_diff_curr = sum_abs_diff_prop;
    }
  }
}

Eigen::VectorXd LapNIGHierarchy::propose_rwmh(
    const Eigen::VectorXd &curr_vals) {
  auto &rng = bayesmix::Rng::Instance().get();
  double candidate_mean =
      curr_vals(0) + stan::math::normal_rng(0, sqrt(hypers->mh_mean_var), rng);
  double candidate_log_scale =
      curr_vals(1) +
      stan::math::normal_rng(0, sqrt(hypers->mh_log_scale_var), rng);
  Eigen::VectorXd proposal(2);
  proposal << candidate_mean, candidate_log_scale;
  return proposal;
}

double LapNIGHierarchy::eval_prior_lpdf_unconstrained(
    Eigen::VectorXd unconstrained_parameters) {
  double mu = unconstrained_parameters(0);
  double log_scale = unconstrained_parameters(1);
  double scale = std::exp(log_scale);
  return stan::math::normal_lpdf(mu, hypers->mean, std::sqrt(hypers->var)) +
         stan::math::inv_gamma_lpdf(scale, hypers->shape, hypers->scale) +
         log_scale;
}

double LapNIGHierarchy::eval_like_lpdf_unconstrained(
    Eigen::VectorXd unconstrained_parameters, bool is_current) {
  double mean = unconstrained_parameters(0);
  double log_scale = unconstrained_parameters(1);
  double scale = std::exp(log_scale);
  double diff_sum = 0;  // Sum of absolute values of data - candidate_mean
  if (is_current) {
    diff_sum = sum_abs_diff_curr;
  } else {
    for (auto &elem : cluster_data_values) {
      diff_sum += std::abs(elem(0, 0) - mean);
    }
    sum_abs_diff_prop = diff_sum;
  }
  return std::log(0.5 / scale) + (-0.5 / scale * diff_sum);
}
