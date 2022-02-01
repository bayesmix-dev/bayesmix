#include "mixture_finite_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "matrix.pb.h"
#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

void MixtureFiniteMixing::update_state(
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

double MixtureFiniteMixing::mass_existing_cluster(
    const unsigned int n, const bool log, const bool propto,
    std::shared_ptr<AbstractHierarchy> hier) const {
  double out;
  double gamma = 1;  //! Has to be a parameter
  if (log) {
    out = std::log(hier->get_card() + gamma);
  } else {
    out = hier->get_card() + gamma;
  }
  return out;
}

double MixtureFiniteMixing::mass_new_cluster(
    const unsigned int n, const bool log, const bool propto,
    const unsigned int n_clust) const {
  double gamma = 1;   //! Has to be a parameter
  double lambda = 5;  //! Has to be a parameter

  if (!check) {
    for (unsigned int t = 0; t < n + 1; ++t) {
      double v = 0;
      for (unsigned int k = 1; k < 1000; ++k) {
        double log_num = std::log(k);
        double log_den = std::log(gamma * k);
        for (unsigned int jj = 1; jj < t; ++jj) {
          log_num += std::log(k - jj);
        }
        for (unsigned int jjj = 1; jjj < n; ++jjj) {
          log_den += std::log(gamma * k + jjj);
        }
        v += std::exp(log_num - log_den + stan::math::poisson_lpmf(k, lambda) +
                      1000);
      }
      std::cout << v << std::endl;
      V.push_back(v);
    }
    check = true;
  }
  double out;
  if (log) {
    out = std::log(V[n_clust + 1] / V[n_clust] * gamma);
  } else {
    out = V[n_clust + 1] / V[n_clust] * gamma;
  }
  return out;
}

void MixtureFiniteMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.totalmass = statecast.mfm_state().totalmass();
  state.logtotmass = std::log(state.totalmass);
}

std::shared_ptr<bayesmix::MixingState> MixtureFiniteMixing::get_state_proto()
    const {
  bayesmix::MFMState state_;
  state_.set_totalmass(state.totalmass);
  auto out = std::make_shared<bayesmix::MixingState>();
  out->mutable_mfm_state()->CopyFrom(state_);
  return out;
}

void MixtureFiniteMixing::initialize_state() {
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
    throw std::invalid_argument("Uunrecognized mixing prior");
  }
}
