#include "mfa_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <cassert>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/rng.h"

double MFAHierarchy::like_lpdf(const Eigen::RowVectorXd& datum) const {
  return stan::math::normal_lpdf(
      datum, state.mu + state.Lambda * state.Lambda.transpose(),
      state.psi.cwiseSqrt());
}

MFA::State MFAHierarchy::draw(const MFA::Hyperparams& params) {
  auto& rng = bayesmix::Rng::Instance().get();
  MFA::State out;

  for (size_t j = 0; j < p; j++) {
    out.mu[j] = stan::math::normal_prec_rng(params.mutilde[j],
                                            1 / sqrt(params.phi), rng);
    out.psi[j] = stan::math::inv_gamma_rng(params.alpha0, params.beta[j], rng);
    for (size_t i = 0; i < params.q; j++) {
      out.Lambda[j, i] = stan::math::normal_prec_rng(0, 1, rng);
    }
  }

  for (size_t i = 0; i < card; i++) {
    for (size_t j = 0; j < params.q; j++) {
      out.Eta[j, i] = stan::math::normal_prec_rng(0, 1, rng);
    }
  }

  return out;
}

void MFAHierarchy::initialize_state() {
  state.mu = hypers->mutilde;
  state.psi = hypers->beta / (hypers->alpha0 + 1.);
  state.Eta = Eigen::MatrixXd::Zero(card, hypers->q);
  state.Lambda = Eigen::MatrixXd::Zero(p, hypers->q);
}

void MFAHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mutilde = bayesmix::to_eigen(prior->fixed_values().mutilde());
    p = hypers->mutilde.size();
    hypers->beta = bayesmix::to_eigen(prior->fixed_values().beta());
    hypers->phi = prior->fixed_values().phi();
    hypers->alpha0 = prior->fixed_values().alpha0();
    hypers->q = prior->fixed_values().q();

    // Check validity
    if (p != hypers->beta.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }

    if (hypers->beta <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->alpha0 <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
    if (phi->beta <= 0) {
      throw std::invalid_argument("Diffusion parameter must be > 0");
    }
    if (phi->q <= 0) {
      throw std::invalid_argument("Number of factors must be > 0");
    }
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void MFAHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState>& states) {
  auto& rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void MFAHierarchy::update_summary_statistics(const Eigen::RowVectorXd& datum,
                                             bool add) {
  if (add) {
    data_sum += datum;
    data.push_back(datum);
  } else {
    data_sum -= datum;
    auto pos = std::find(data.begin(), data.end(), datum);
    data.erase(pos);
  }
}

void MFAHierarchy::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(p);
  data.clear();
}

void MFAHierarchy::set_state_from_proto(
    const google::protobuf::Message& state_) {
  auto& statecast = downcast_state(state_);
  state.mu = statecast.mfa_state().mu();
  state.psi = statecast.mfa_state().psi();
  state.Eta = statecast.mfa_state().Eta();
  state.Lambda = statecast.mfa_state().Lambda();
  set_card(statecast.cardinality());
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
MFAHierarchy::get_state_proto() const {
  bayesmix::MFAState state_;
  state_.set_mu(state.mu);
  state_.set_psi(state.psi);
  state_.set_Eta(state.Eta);
  state_.set_Lambda(state.Lambda);

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_mfa_state()->CopyFrom(state_);
  return out;
}

void MFAHierarchy::set_hypers_from_proto(
    const google::protobuf::Message& hypers_) {
  auto& hyperscast = downcast_hypers(hypers_).mfa_state();
  hypers->mutilde = hyperscast.mutilde();
  hypers->alpha0 = hyperscast.alpha0();
  hypers->beta = hyperscast.beta();
  hypers->phi = hyperscast.phi();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
MFAHierarchy::get_hypers_proto() const {
  bayesmix::MFAPrior hypers_;
  hypers_.set_mutilde(hypers->mutilde);
  hypers_.set_alpha0(hypers->alpha0);
  hypers_.set_beta(hypers->beta);
  hypers_.set_phi(hypers->phi);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_mfa_state()->CopyFrom(hypers_);
  return out;
}

void MFAHierarchy::sample_full_cond(bool update_params = false) override {
  assert(update_params != true);  // should never be true
  if (this->card == 0) {
    // No posterior update possible
    static_cast<Derived*>(this)->sample_prior();
  } else {
    sample_Eta();
    sample_mu();
    sample_psi();
    sample_Lambda();
  }
}

void MFAHierarchy::sample_Eta() const {
  auto& rng = bayesmix::Rng::Instance().get();

  Eigen::MatrixXd Sigmaeta =
      (Eigen::MatrixXd::Identity(hypers->q, q) +
       state.Lambda.transpose() *
           Eigen::MatrixXd(psi.cwiseInverse()).asDiagonal() * state.Lambda)
          .inverse();

  for (size_t i = 0; i < card; i++) {
    state.Eta.row(i) = stan::math::multi_normal_rng(
        Sigmaeta * state.Lambda.transpose() *
            Eigen::MatrixXd(psi.cwiseInverse()).asDiagonal() *
            (data[i] - state.mu),
        Sigmaeta, rng);
  }
}

void MFAHierarchy::sample_mu() const {
  auto& rng = bayesmix::Rng::Instance().get();

  Eigen::MatrixXd Sigmamu =
      (hypers->phi * Eigen::MatrixXd::Identity(p, p) +
       card * Eigen::MatrixXd(psi.cwiseInverse()).asDiagonal())
          .inverse();

  Eigen::VectorXd Somma = Eigen::VectorXd::Zero(p);

  for (size_t i = 0; i < card; i++) {
    Somma += state.Lambda * state.Eta.row(i);
  }

  state.mu = stan::math::multi_normal_rng(
      Sigmamu * (hypers->phi * hypers->mutilde +
                 Eigen::MatrixXd(psi.cwiseInverse()).asDiagonal() *
                     (data_sum - Somma)),
      Sigmamu, rng);
}

void MFAHierarchy::sample_Lambda() const {
  auto& rng = bayesmix::Rng::Instance().get();

  for (size_t j = 0; j < p; j++) {
    Eigen::MatrixXd Sigmalambda =
        (Eigen::MatrixXd::Identity(hypers->q, q) +
         state.Eta.transpose() / state.psi[j] * state.Eta)
            .inverse();

    state.Lambda.row(j) = stan::math::multi_normal_rng(
        Sigmalambda * state.Eta.transpose() / state.psi[j] *
            (data.col(j) - state.mu[j]),
        Sigmalambda, rng);
  }
}

void MFAHierarchy::sample_psi() const {
  auto& rng = bayesmix::Rng::Instance().get();

  for (size_t j = 0; j < p; j++) {
    double S = 0;
    for (size_t i = 0; i < card; i++) {
      S += std::pow((data[i, j] - state.mu[j] -
                     state.Lambda.row(j).dot(state.Eta.row(i))),
                    2);
    }
    state.psi[j] = stan::math::inv_gamma_rng(hypers->alpha0 + card / 2,
                                             hypers->beta[j] + S / 2, rng);
  }
}
