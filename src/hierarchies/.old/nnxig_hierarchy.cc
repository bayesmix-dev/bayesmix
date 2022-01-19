#include "nnxig_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/rng.h"

double NNxIGHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::normal_lpdf(datum(0), state.mean, sqrt(state.var));
}

void NNxIGHierarchy::initialize_state() {
  state.mean = hypers->mean;
  state.var = hypers->scale / (hypers->shape + 1);
}

void NNxIGHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = prior->fixed_values().mean();
    hypers->var = prior->fixed_values().var();
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();

    // Check validity
    if (hypers->var <= 0) {
      throw std::invalid_argument("Variance parameter must be > 0");
    }
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("scale parameter must be > 0");
    }
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void NNxIGHierarchy::update_summary_statistics(const Eigen::RowVectorXd &datum,
                                               bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
}

void NNxIGHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void NNxIGHierarchy::clear_summary_statistics() {
  data_sum = 0;
  data_sum_squares = 0;
}

void NNxIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.var = statecast.uni_ls_state().var();
  set_card(statecast.cardinality());
}

void NNxIGHierarchy::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).nnxig_state();
  hypers->mean = hyperscast.mean();
  hypers->var = hyperscast.var();
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
NNxIGHierarchy::get_state_proto() const {
  bayesmix::UniLSState state_;
  state_.set_mean(state.mean);
  state_.set_var(state.var);

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_uni_ls_state()->CopyFrom(state_);
  return out;
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
NNxIGHierarchy::get_hypers_proto() const {
  bayesmix::NxIGDistribution hypers_;
  hypers_.set_mean(hypers->mean);
  hypers_.set_var(hypers->var);
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_nnxig_state()->CopyFrom(hypers_);
  return out;
}

void NNxIGHierarchy::sample_full_cond(bool update_params) {
  if (this->card == 0) {
    // No posterior update possible
    sample_prior();
  } else {
    NNxIG::Hyperparams params =
        update_params ? compute_posterior_hypers() : posterior_hypers;
    state = draw(params);
  }
}

NNxIG::State NNxIGHierarchy::draw(const NNxIG::Hyperparams &params) {
  auto &rng = bayesmix::Rng::Instance().get();
  NNxIG::State out;
  out.var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  out.mean = stan::math::normal_rng(params.mean, sqrt(params.var), rng);
  return out;
}

NNxIG::Hyperparams NNxIGHierarchy::compute_posterior_hypers() const {
  // Initialize relevant variables
  if (card == 0) {  // no update possible
    return *hypers;
  }
  // Compute posterior hyperparameters
  NNxIG::Hyperparams post_params;
  double var_y = data_sum_squares - 2 * state.mean * data_sum +
                 card * state.mean * state.mean;
  post_params.mean = (hypers->var * data_sum + state.var * hypers->mean) /
                     (card * hypers->var + state.var);
  post_params.var =
      (state.var * hypers->var) / (card * hypers->var + state.var);
  post_params.shape = hypers->shape + 0.5 * card;
  post_params.scale = hypers->scale + 0.5 * var_y;
  return post_params;
}

void NNxIGHierarchy::save_posterior_hypers() {
  posterior_hypers = compute_posterior_hypers();
}
