#include "dirichlet_mixing.hpp"

#include <google/protobuf/stubs/casts.h>

#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/mixing_prior.pb.h"
#include "../../proto/cpp/mixing_state.pb.h"
#include "../utils/rng.hpp"

void DirichletMixing::update_hypers(
    const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
    unsigned int n) {
  if (prior.has_fixed_value()) {
    return;
  } else if (prior.has_gamma_prior()) {
    auto &rng = bayesmix::Rng::Instance().get();

    // Recover parameters
    unsigned int k = unique_values.size();
    double alpha = prior.gamma_prior().shape();
    double beta = prior.gamma_prior().rate();

    double phi = stan::math::gamma_rng(totalmass + 1, n, rng);
    double odds = (alpha + k - 1) / (n * (beta - log(phi)));
    double prob = odds / (1 + odds);
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p <= prob) {
      totalmass = stan::math::gamma_rng(alpha + k, beta - log(phi), rng);
    } else {
      totalmass = stan::math::gamma_rng(alpha + k - 1, beta - log(phi), rng);
    }
  } else {
    std::invalid_argument("Error: no possible valid update");
  }
}

void DirichletMixing::set_prior(const google::protobuf::Message &prior_) {
  const bayesmix::DPPrior &currcast =
      google::protobuf::internal::down_cast<const bayesmix::DPPrior &>(prior_);
  prior = currcast;
  if (prior.has_fixed_value()) {
    totalmass = prior.fixed_value().value();
    assert(totalmass > 0);
  } else if (prior.has_gamma_prior()) {
    assert(prior.gamma_prior().shape() > 0);
    assert(prior.gamma_prior().rate() > 0);
    totalmass = prior.gamma_prior().shape() / prior.gamma_prior().rate();
  } else {
    std::invalid_argument("Error: argument proto is not appropriate");
  }
}

void DirichletMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::DPState state;
  state.set_totalmass(totalmass);

  google::protobuf::internal::down_cast<bayesmix::MixingState *>(out)
      ->mutable_dp_state()
      ->CopyFrom(state);
}
