#include "truncated_sb_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <numeric>
#include <stan/math/prim.hpp>
#include <vector>

#include "logit_sb_mixing.h"
#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"


void TruncatedSBMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  assert(unique_values.size() == num_components);
  // Initialize relevant objects
  auto &rng = bayesmix::Rng::Instance().get();
  auto priorcast = cast_prior();
  // Recover cluster cardinalities
  std::vector<unsigned int> cards(num_components);
  for (int i = 0; i < num_components; i++) {
    cards[i] = unique_values[i]->get_card();
  }
  // Get prior parameters
  Eigen::MatrixXd shapes = get_prior_shape_parameters();
  // Loop over components
  for (int i = 0; i < num_components - 1; i++) {
    // Count data points in subsequent clusters than the i-th one
    unsigned int subseq_count = 0;
    for (int j = i + 1; j < num_components; j++) {
      subseq_count += cards[j];
    }
    // Draw new value for i-th stick
    state.sticks(i) = stan::math::beta_rng(shapes(0, i) + cards[i],
                                           shapes(1, i) + subseq_count, rng);
  }
  state.sticks(num_components - 1) = 1.0;
  // Update logweights
  state.logweights = logweights_from_sticks();
}

Eigen::VectorXd TruncatedSBMixing::get_weights(
    const bool log, const bool propto,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  if (log) {
    return state.logweights;
  } else {
    return state.logweights.array().exp();
  }
}

void TruncatedSBMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.sticks = bayesmix::to_eigen(statecast.trunc_sb_state().sticks());
  state.logweights = logweights_from_sticks();
}

std::shared_ptr<bayesmix::MixingState> TruncatedSBMixing::get_state_proto() 
    const {
  bayesmix::TruncSBState state_;
  bayesmix::to_proto(state.sticks, state_.mutable_sticks());
  bayesmix::to_proto(state.logweights, state_.mutable_logweights());

  auto out = std::make_unique<bayesmix::MixingState>();
  out->mutable_trunc_sb_state()->CopyFrom(state_);
  return out;
}

void TruncatedSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  num_components = priorcast->num_components();
  state.sticks = Eigen::VectorXd(num_components);
  if (priorcast->has_beta_priors()) {
    if (priorcast->beta_priors().beta_distributions_size() != num_components) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    for (int i = 0; i < num_components - 1; i++) {
      if (priorcast->beta_priors().beta_distributions(i).shape_a() <= 0 or
          priorcast->beta_priors().beta_distributions(i).shape_b() <= 0) {
        throw std::invalid_argument("Beta shape parameters must be > 0");
      }
    }
  } else if (priorcast->has_dp_prior()) {
    if (priorcast->dp_prior().totalmass() <= 0) {
      throw std::invalid_argument("Total mass parameter must be > 0");
    }
  } else if (priorcast->has_py_prior()) {
    if (priorcast->py_prior().strength() <=
        -priorcast->py_prior().discount()) {
      throw std::invalid_argument("Prior parameters are not valid");
    }
    if (priorcast->py_prior().discount() < 0 or
        priorcast->py_prior().discount() >= 1) {
      throw std::invalid_argument("Discount parameter must be in [0,1)");
    }
  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
  // Initialize sticks
  Eigen::MatrixXd shapes = get_prior_shape_parameters();
  for (int i = 0; i < num_components - 1; i++) {
    state.sticks(i) = shapes(0, i) / (shapes(0, i) + shapes(1, i));
  }
  state.sticks(num_components - 1) = 1.0;
  // Initialize logweights
  state.logweights = logweights_from_sticks();
}

Eigen::VectorXd TruncatedSBMixing::logweights_from_sticks() const {
  // Compute cumulative sums of logarithms
  std::vector<double> cumsum(num_components + 1, 0.0);
  for (int h = 1; h < num_components + 1; h++) {
    cumsum[h] = cumsum[h - 1] + std::log(1 - state.sticks(h - 1));
  }
  // Compute weights
  Eigen::VectorXd logweights(num_components);
  for (int h = 0; h < num_components; h++) {
    logweights(h) = std::log(state.sticks(h)) + cumsum[h];
  }
  return logweights;
}

Eigen::MatrixXd TruncatedSBMixing::get_prior_shape_parameters() const {
  Eigen::MatrixXd shapes(2, num_components);
  auto priorcast = cast_prior();
  if (priorcast->has_beta_priors()) {
    // Individual shape parameters for each component
    for (int i = 0; i < num_components; i++) {
      shapes(0, i) = priorcast->beta_priors().beta_distributions(i).shape_a();
      shapes(1, i) = priorcast->beta_priors().beta_distributions(i).shape_b();
    }
  } else if (priorcast->has_dp_prior()) {
    // Uniform shape parameters for all components, computed from DP parameters
    double totmass = priorcast->dp_prior().totalmass();
    shapes.row(0) = Eigen::VectorXd::Ones(num_components);
    shapes.row(1) = totmass * Eigen::VectorXd::Ones(num_components);
  } else if (priorcast->has_py_prior()) {
    // Uniform shape parameters for all components, computed from PY parameters
    double strength = priorcast->py_prior().strength();
    double disc = priorcast->py_prior().discount();
    for (int i = 0; i < num_components; i++) {
      shapes(0, i) = 1.0 - disc;
      shapes(1, i) = strength + i * disc;
    }
  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
  return shapes;
}
