#include "dirichlet_mixing.hpp"

#include <google/protobuf/stubs/casts.h>

#include <cassert>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/mixing_prior.pb.h"
#include "../../proto/cpp/mixing_state.pb.h"
#include "../utils/rng.hpp"

void DirichletMixing::initialize() {
  assert(prior != nullptr && "Error: prior was not provided");
}

void DirichletMixing::update_state(
    const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
    unsigned int n) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_value()) {
    return;
  }

  else if (prior->has_gamma_prior()) {
    // Recover parameters
    unsigned int k = unique_values.size();
    double alpha = prior->gamma_prior().totalmass_prior().shape();
    double beta = prior->gamma_prior().totalmass_prior().rate();

    double phi = stan::math::gamma_rng(state.totalmass + 1, n, rng);
    double odds = (alpha + k - 1) / (n * (beta - log(phi)));
    double prob = odds / (1 + odds);
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p <= prob) {
      state.totalmass = stan::math::gamma_rng(alpha + k, beta - log(phi), rng);
    } else {
      state.totalmass = stan::math::gamma_rng(alpha + k - 1, beta - log(phi), rng);
    }
  }

  else {
    throw std::invalid_argument("Error: unrecognized prior");
  }
}

void DirichletMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::DPState &>(state_);
  state.totalmass = statecast.totalmass();
}

void DirichletMixing::set_prior(const google::protobuf::Message &prior_) {
  auto &priorcast =
      google::protobuf::internal::down_cast<const bayesmix::DPPrior &>(prior_);
  prior = std::make_shared<bayesmix::DPPrior>(priorcast);
  if (prior->has_fixed_value()) {
    state.totalmass = prior->fixed_value().totalmass();
    assert(state.totalmass > 0);
  }

  else if (prior->has_gamma_prior()) {
    double alpha = prior->gamma_prior().totalmass_prior().shape();
    double beta = prior->gamma_prior().totalmass_prior().rate();
    assert(alpha > 0);
    assert(beta > 0);
    state.totalmass = alpha / beta;
  }

  else {
    throw std::invalid_argument("Error: unrecognized prior");
  }
}

void DirichletMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::DPState state_;
  state_.set_totalmass(state.totalmass);

  google::protobuf::internal::down_cast<bayesmix::MixingState *>(out)
      ->mutable_dp_state()
      ->CopyFrom(state_);
}
