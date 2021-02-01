#include "pityor_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <cassert>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"

void PitYorMixing::initialize() {
  assert(prior != nullptr && "Error: prior was not provided");
}

//! \param card Cardinality of the cluster
//! \param n    Total number of data points
//! \return     Probability value
double PitYorMixing::mass_existing_cluster(std::shared_ptr<BaseHierarchy> hier,
                                           const unsigned int n, bool log,
                                           bool propto) const {
  double out;
  if (hier->get_card() == 0) {
    out = 0;
  } else {
    out = (hier->get_card() - state.discount) / (n + state.strength);
  }
  if (log) out = std::log(out);
  return out;
}

//! \param n_clust Number of clusters
//! \param n       Total number of data points
//! \return        Probability value
double PitYorMixing::mass_new_cluster(const unsigned int n_clust,
                                      const unsigned int n, bool log,
                                      bool propto) const {
  double out =
      (state.strength + state.discount * n_clust) / (n + state.strength);
  if (log) out = std::log(out);
  return out;
}

void PitYorMixing::update_state(
    const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
    unsigned int n) {
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Error: unrecognized prior");
  }
}

void PitYorMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::PYState &>(state_);
  state.strength = statecast.strength();
  state.discount = statecast.discount();
}

void PitYorMixing::set_prior(const google::protobuf::Message &prior_) {
  auto &priorcast =
      google::protobuf::internal::down_cast<const bayesmix::PYPrior &>(prior_);
  prior = std::make_shared<bayesmix::PYPrior>(priorcast);
  if (prior->has_fixed_values()) {
    state.strength = prior->fixed_values().strength();
    state.discount = prior->fixed_values().discount();
    assert(state.strength > -state.discount);
    assert(0 <= state.discount && state.discount < 1);

  } else {
    throw std::invalid_argument("Error: unrecognized prior");
  }
}

void PitYorMixing::write_state_to_proto(google::protobuf::Message *out) const {
  bayesmix::PYState state_;
  state_.set_strength(state.strength);
  state_.set_discount(state.discount);

  google::protobuf::internal::down_cast<bayesmix::PYState *>(out)->CopyFrom(
      state_);
}
