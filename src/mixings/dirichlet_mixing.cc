#include "dirichlet_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <stan/math/prim/prob.hpp>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/utils/rng.h"

void DirichletMixing::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
}

//! \param card Cardinality of the cluster
//! \param n    Total number of data points
//! \return     Probability value
double DirichletMixing::mass_existing_cluster(
    const unsigned int n, const bool log, const bool propto,
    std::shared_ptr<BaseHierarchy> hier,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
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

//! \param n_clust Number of clusters
//! \param n       Total number of data points
//! \return        Probability value
double DirichletMixing::mass_new_cluster(
    const unsigned int n, const bool log, const bool propto,
    const unsigned int n_clust,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
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
      state.totalmass =
          stan::math::gamma_rng(alpha + k - 1, beta - log(phi), rng);
    }
  }

  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

void DirichletMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::DPState &>(state_);
  state.totalmass = statecast.totalmass();
  state.logtotmass = std::log(state.totalmass);
}

void DirichletMixing::set_prior(const google::protobuf::Message &prior_) {
  auto &priorcast =
      google::protobuf::internal::down_cast<const bayesmix::DPPrior &>(prior_);
  prior = std::make_shared<bayesmix::DPPrior>(priorcast);
  if (prior->has_fixed_value()) {
    state.totalmass = prior->fixed_value().totalmass();
    if (state.totalmass <= 0) {
      throw std::invalid_argument("Total mass parameter must be > 0");
    }
  }

  else if (prior->has_gamma_prior()) {
    double alpha = prior->gamma_prior().totalmass_prior().shape();
    double beta = prior->gamma_prior().totalmass_prior().rate();
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

void DirichletMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::DPState state_;
  state_.set_totalmass(state.totalmass);

  google::protobuf::internal::down_cast<bayesmix::MixingState *>(out)
      ->mutable_dp_state()
      ->CopyFrom(state_);
}
