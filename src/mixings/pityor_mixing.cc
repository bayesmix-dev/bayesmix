#include "pityor_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

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

double PitYorMixing::mass_existing_cluster(
    const unsigned int n, const bool log, const bool propto,
    std::shared_ptr<AbstractHierarchy> hier,
    const unsigned int n_clust) const {
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

double PitYorMixing::mass_new_cluster(const unsigned int n, const bool log,
                                      const bool propto,
                                      const unsigned int n_clust) const {
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

void PitYorMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.strength = statecast.py_state().strength();
  state.discount = statecast.py_state().discount();
}

std::shared_ptr<bayesmix::MixingState> PitYorMixing::get_state_proto() const {
  bayesmix::PYState state_;
  state_.set_strength(state.strength);
  state_.set_discount(state.discount);

  auto out = std::make_shared<bayesmix::MixingState>();
  out->mutable_py_state()->CopyFrom(state_);
  return out;
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
