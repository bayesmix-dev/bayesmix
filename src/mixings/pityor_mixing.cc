#include "pityor_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

void PitYorMixing::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  initialize_state();
}

//! @param card Cardinality of the cluster
//! @param n    Total number of data points
//! @return     Probability value
double PitYorMixing::mass_existing_cluster(
    const unsigned int n, const bool log, const bool propto,
    std::shared_ptr<AbstractHierarchy> hier,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  double out;
  if (hier->get_card() == 0) {
    return log ? stan::math::NEGATIVE_INFTY : 0;
  }
  if (log) {
    out = std::log(hier->get_card() - state.discount);
    if (!propto) out -= std::log(n + state.strength);
  } else {
    out = hier->get_card() - state.discount;
    if (!propto) out /= n + state.strength;
  }
  return out;
}

//! @param n_clust Number of clusters
//! @param n       Total number of data points
//! @return        Probability value
double PitYorMixing::mass_new_cluster(
    const unsigned int n, const bool log, const bool propto,
    const unsigned int n_clust,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  double out;
  if (log) {
    out = std::log(state.strength + state.discount * n_clust);
    if (!propto) out -= std::log(n + state.strength);
  } else {
    out = state.strength + state.discount * n_clust;
    if (!propto) out /= n + state.strength;
  }
  return out;
}

void PitYorMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

void PitYorMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::MixingState &>(
          state_);
  state.strength = statecast.py_state().strength();
  state.discount = statecast.py_state().discount();
}

void PitYorMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_values()) {
    state.strength = priorcast->fixed_values().strength();
    state.discount = priorcast->fixed_values().discount();
    if (state.strength <= -state.discount) {
      throw std::invalid_argument("Mixing parameters are not valid");
    }
    if (state.discount < 0 or state.discount >= 1) {
      throw std::invalid_argument("Discount parameter must be in [0,1)");
    }

  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

void PitYorMixing::write_state_to_proto(google::protobuf::Message *out) const {
  bayesmix::PYState state_;
  state_.set_strength(state.strength);
  state_.set_discount(state.discount);

  auto *out_cast =
      google::protobuf::internal::down_cast<bayesmix::MixingState *>(out);
  out_cast->mutable_py_state()->CopyFrom(state_);
}
