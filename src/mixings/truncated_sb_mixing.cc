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

void TruncatedSBMixing::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  auto priorcast = cast_prior();
  num_components = priorcast->num_components();
  initialize_state();
}

void TruncatedSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  state.sticks = Eigen::VectorXd(num_components);
  if (priorcast->has_beta_priors()) {
    if (priorcast->beta_priors().beta_distributions_size() != num_components) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    for (int i = 0; i < num_components - 1; i++) {
      double shape_a =
          priorcast->beta_priors().beta_distributions(i).shape_a();
      double shape_b =
          priorcast->beta_priors().beta_distributions(i).shape_b();
      if (shape_a <= 0 or shape_b <= 0) {
        throw std::invalid_argument("Beta shape parameters must be > 0");
      }
      state.sticks(i) = shape_a / (shape_a + shape_b);
    }
    state.sticks(num_components - 1) = 1.0;

  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

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
  // Loop over components
  for (int i = 0; i < num_components - 1; i++) {
    // Get prior parameters
    double shape_a = priorcast->beta_priors().beta_distributions(i).shape_a();
    double shape_b = priorcast->beta_priors().beta_distributions(i).shape_b();
    // Count data points in subsequent clusters than the i-th one
    unsigned int subseq_count = 0;
    for (int j = i + 1; i < num_components; j++) {
      subseq_count += cards[j];
    }
    // Draw new value for i-th stick
    state.sticks(i) =
        stan::math::beta_rng(shape_a + cards[i], shape_b + subseq_count, rng);
  }
  state.sticks(num_components - 1) = 1.0;
}

void TruncatedSBMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::MixingState &>(
          state_);
  state.sticks = bayesmix::to_eigen(statecast.trunc_sb_state().sticks());
}

void TruncatedSBMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::TruncSBState state_;
  bayesmix::to_proto(state.sticks, state_.mutable_sticks());
  google::protobuf::internal::down_cast<bayesmix::MixingState *>(out)
      ->mutable_trunc_sb_state()
      ->CopyFrom(state_);
}

Eigen::VectorXd TruncatedSBMixing::get_weights(
    const bool log, const bool propto,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  // Compute cumulative sums of logarithms
  std::vector<double> cumsum(num_components + 1, 0.0);
  for (int h = 1; h < num_components + 1; h++) {
    cumsum[h] = cumsum[h - 1] + std::log(state.sticks(h - 1));
  }
  // Compute weights
  Eigen::VectorXd logweights(num_components);
  for (int h = 0; h < num_components; h++) {
    logweights(h) = std::log(state.sticks(h)) + cumsum[h];
  }
  if (log) {
    return logweights;
  } else {
    return logweights.array().exp();
  }
}
