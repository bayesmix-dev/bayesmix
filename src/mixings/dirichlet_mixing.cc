#include "dirichlet_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/rng.h"

void DirichletMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  auto &rng = bayesmix::Rng::Instance().get();
  auto priorcast = cast_prior();
  unsigned int n = allocations.size();

  if (priorcast->has_fixed_value()) {
    return;
  }

  else if (priorcast->has_gamma_prior()) {
    // Recover parameters
    unsigned int k = unique_values.size();
    double alpha = priorcast->gamma_prior().totalmass_prior().shape();
    double beta = priorcast->gamma_prior().totalmass_prior().rate();
    // Update state (see Neal (2000) for details)
    double phi = stan::math::gamma_rng(state.totalmass + 1, n, rng);
    double odds = (alpha + k - 1) / (n * (beta - log(phi)));
    double prob = odds / (1 + odds);
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p <= prob) {
      state.totalmass = stan::math::gamma_rng(alpha + k, beta - log(phi), rng);
    } else {
      state.totalmass =
          stan::math::gamma_rng(alpha + k - 1, beta - log(phi), rng);
    }
  }

  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

double DirichletMixing::mass_existing_cluster(
    const unsigned int n, const bool log, const bool propto,
    std::shared_ptr<AbstractHierarchy> hier,
    const unsigned int n_clust) const {
  double out;
  if (log) {
    out = hier->get_log_card();
    if (!propto) out -= std::log(n + state.totalmass);
  } else {
    out = 1.0 * hier->get_card();
    if (!propto) out /= (n + state.totalmass);
  }
  return out;
}

double DirichletMixing::mass_new_cluster(const unsigned int n, const bool log,
                                         const bool propto,
                                         const unsigned int n_clust) const {
  double out;
  if (log) {
    out = state.logtotmass;
    if (!propto) out -= std::log(n + state.totalmass);
  } else {
    out = state.totalmass;
    if (!propto) out /= (n + state.totalmass);
  }
  return out;
}

void DirichletMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.totalmass = statecast.dp_state().totalmass();
  state.logtotmass = std::log(state.totalmass);
}

std::shared_ptr<bayesmix::MixingState> DirichletMixing::get_state_proto()
    const {
  bayesmix::DPState state_;
  state_.set_totalmass(state.totalmass);
  auto out = std::make_shared<bayesmix::MixingState>();
  out->mutable_dp_state()->CopyFrom(state_);
  return out;
}

void DirichletMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_value()) {
    state.totalmass = priorcast->fixed_value().totalmass();
    if (state.totalmass <= 0) {
      throw std::invalid_argument("Total mass parameter must be > 0");
    }
  }

  else if (priorcast->has_gamma_prior()) {
    double alpha = priorcast->gamma_prior().totalmass_prior().shape();
    double beta = priorcast->gamma_prior().totalmass_prior().rate();
    if (alpha <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (beta <= 0) {
      throw std::invalid_argument("Rate parameter must be > 0");
    }
    state.totalmass = alpha / beta;
  }

  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}
