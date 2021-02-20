#include "pityor_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"

void PitYorMixing::initialize(const unsigned int n_clust /*= 1*/) {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  initialize_state();
}

//! \param card Cardinality of the cluster
//! \param n    Total number of data points
//! \return     Probability value
double PitYorMixing::mass_existing_cluster(
    const unsigned int n, const bool log, const bool propto,
    std::shared_ptr<AbstractHierarchy> hier,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
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
double PitYorMixing::mass_new_cluster(
    const unsigned int n, const bool log, const bool propto,
    const unsigned int n_clust,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  double out =
      (state.strength + state.discount * n_clust) / (n + state.strength);
  if (log) out = std::log(out);
  return out;
}

void PitYorMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations, const unsigned int n) {
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
      google::protobuf::internal::down_cast<const bayesmix::PYState &>(state_);
  state.strength = statecast.strength();
  state.discount = statecast.discount();
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

  google::protobuf::internal::down_cast<bayesmix::PYState *>(out)->CopyFrom(
      state_);
}
