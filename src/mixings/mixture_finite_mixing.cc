#include "mixture_finite_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/rng.h"

void MixtureFiniteMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  auto &rng = bayesmix::Rng::Instance().get();
  auto priorcast = cast_prior();
  unsigned int n = allocations.size();

  if (priorcast->has_fixed_value()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

double MixtureFiniteMixing::mass_existing_cluster(
    const unsigned int n, const unsigned int n_clust, const bool log,
    const bool propto, std::shared_ptr<AbstractHierarchy> hier) const {
  double out;
  double V1 = get_V_t(n_clust, n);
  double V2 = get_V_t(n_clust + 1, n);

  if (log) {
    out = std::log(hier->get_card() + state.gamma);
    if (!propto) {
      out -= std::log(n + n_clust * state.gamma + (V2 / V1 * state.gamma));
    }
  } else {
    out = hier->get_card() + state.gamma;
    if (!propto) {
      out = out / (n + n_clust * state.gamma + (V2 / V1 * state.gamma));
    }
  }
  return out;
}

double MixtureFiniteMixing::mass_new_cluster(const unsigned int n,
                                             const unsigned int n_clust,
                                             const bool log,
                                             const bool propto) const {
  double out;
  double V1 = get_V_t(n_clust, n);
  double V2 = get_V_t(n_clust + 1, n);

  if (log) {
    out = std::log(V2 / V1 * state.gamma);
    if (!propto) {
      out -= std::log(n + n_clust * state.gamma + V2 / V1 * state.gamma);
    }
  } else {
    out = V2 / V1 * state.gamma;
    if (!propto) {
      out = out / (n + n_clust * state.gamma + V2 / V1 * state.gamma);
    }
  }
  return out;
}

void MixtureFiniteMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.lambda = statecast.mfm_state().lambda();
  state.gamma = statecast.mfm_state().gamma();
}

std::shared_ptr<bayesmix::MixingState> MixtureFiniteMixing::get_state_proto()
    const {
  bayesmix::MFMState state_;
  state_.set_lambda(state.lambda);
  state_.set_gamma(state.gamma);
  auto out = std::make_shared<bayesmix::MixingState>();
  out->mutable_mfm_state()->CopyFrom(state_);
  return out;
}

void MixtureFiniteMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_value()) {
    state.lambda = priorcast->fixed_value().lambda();
    state.gamma = priorcast->fixed_value().gamma();
    if (state.lambda <= 0) {
      throw std::invalid_argument("Poisson rate lambda must be > 0");
    }
    if (state.gamma <= 0) {
      throw std::invalid_argument("Dirichlet parameter gamma must be > 0");
    }
  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

void MixtureFiniteMixing::init_V_and_C(unsigned int n) const {
  // Initialize V
  V = std::vector<double>(n, -1);

  // The values of V decrease fast enough to become smaller than the smallest
  // number that can be contained in double, in order to avoid that introduce
  // C as the first term of the series and multiply all the
  // elements of the series with exp(-C)

  double log_den = std::log(state.gamma);
  for (unsigned int i = 1; i < n; ++i) {
    log_den += std::log(state.gamma * 1 + i);
  }
  C = -log_den + stan::math::poisson_lpmf(0, state.lambda);

  V_C_are_initialized = true;
}

void MixtureFiniteMixing::compute_V_t(double t, unsigned int n) const {
  double sum = 0;
  unsigned int k = 1;
  double last_term_sum_rate =
      1;  // Rate between the last computed term of the series
          // and the sum computed up to that point. This value
          // is used as a stopping criterion
  while (last_term_sum_rate > 1e-8) {
    double log_num = std::log(k);
    double log_den = std::log(state.gamma * k);

    for (unsigned int i = 1; i < t; ++i) {
      log_num += std::log(k - i);
    }
    for (unsigned int i = 1; i < n; ++i) {
      log_den += std::log(state.gamma * k + i);
    }

    // Update the last_term_sum_rate
    if (sum == 0) {
      last_term_sum_rate = 1;
    } else {
      last_term_sum_rate =
          std::exp(log_num - log_den +
                   stan::math::poisson_lpmf(k - 1, state.lambda) - C) /
          sum;
    }

    sum += std::exp(log_num - log_den +
                    stan::math::poisson_lpmf(k - 1, state.lambda) - C);
    ++k;
  }
  V[t] = sum;
}

double MixtureFiniteMixing::get_V_t(double t, unsigned int n) const {
  if (!V_C_are_initialized) {
    init_V_and_C(n + 1);
  }
  if (V[t] < 0) {
    compute_V_t(t, n + 1);
  }
  return V[t];
}
