#include "fa_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

double FAHierarchy::like_lpdf(const Eigen::RowVectorXd& datum) const {
  return bayesmix::multi_normal_lpdf_woodbury_chol(
      datum, state.mu, state.psi_inverse, state.cov_wood, state.cov_logdet);
}

FA::State FAHierarchy::draw(const FA::Hyperparams& params) {
  auto& rng = bayesmix::Rng::Instance().get();
  FA::State out;
  out.mu = params.mutilde;
  out.psi = params.beta / (params.alpha0 + 1.);
  out.eta = Eigen::MatrixXd::Zero(card, params.q);
  out.lambda = Eigen::MatrixXd::Zero(dim, params.q);

  for (size_t j = 0; j < dim; j++) {
    out.mu[j] =
        stan::math::normal_rng(params.mutilde[j], sqrt(params.phi), rng);

    out.psi[j] = stan::math::inv_gamma_rng(params.alpha0, params.beta[j], rng);

    for (size_t i = 0; i < params.q; i++) {
      out.lambda(j, i) = stan::math::normal_rng(0, 1, rng);
    }
  }

  for (size_t i = 0; i < card; i++) {
    for (size_t j = 0; j < params.q; j++) {
      out.eta(i, j) = stan::math::normal_rng(0, 1, rng);
    }
  }

  out.psi_inverse = out.psi.cwiseInverse().asDiagonal();
  compute_wood_factors(out.cov_wood, out.cov_logdet, out.lambda,
                       out.psi_inverse);

  return out;
}

void FAHierarchy::initialize_state() {
  state.mu = hypers->mutilde;
  state.psi = hypers->beta / (hypers->alpha0 + 1.);
  state.eta = Eigen::MatrixXd::Zero(card, hypers->q);
  state.lambda = Eigen::MatrixXd::Zero(dim, hypers->q);
  state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
  compute_wood_factors(state.cov_wood, state.cov_logdet, state.lambda,
                       state.psi_inverse);
}

void FAHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mutilde = bayesmix::to_eigen(prior->fixed_values().mutilde());
    dim = hypers->mutilde.size();
    hypers->beta = bayesmix::to_eigen(prior->fixed_values().beta());
    hypers->phi = prior->fixed_values().phi();
    hypers->alpha0 = prior->fixed_values().alpha0();
    hypers->q = prior->fixed_values().q();

    // Automatic initialization
    if (dim == 0) {
      hypers->mutilde = dataset_ptr->colwise().mean();
      dim = hypers->mutilde.size();
    }
    if (hypers->beta.size() == 0) {
      Eigen::MatrixXd centered =
          dataset_ptr->rowwise() - dataset_ptr->colwise().mean();
      auto cov_llt = ((centered.transpose() * centered) /
                      double(dataset_ptr->rows() - 1.))
                         .llt();
      Eigen::MatrixXd precision_matrix(
          cov_llt.solve(Eigen::MatrixXd::Identity(dim, dim)));
      hypers->beta =
          (hypers->alpha0 - 1) * precision_matrix.diagonal().cwiseInverse();
      if (hypers->alpha0 == 1) {
        throw std::invalid_argument(
            "Scale parameter must be different than 1 when automatic "
            "initialization is used");
      }
    }
    // Check validity
    if (dim != hypers->beta.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    for (size_t j = 0; j < dim; j++) {
      if (hypers->beta[j] <= 0) {
        throw std::invalid_argument("Shape parameter must be > 0");
      }
    }
    if (hypers->alpha0 <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
    if (hypers->phi <= 0) {
      throw std::invalid_argument("Diffusion parameter must be > 0");
    }
    if (hypers->q <= 0) {
      throw std::invalid_argument("Number of factors must be > 0");
    }
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void FAHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState>& states) {
  auto& rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void FAHierarchy::update_summary_statistics(const Eigen::RowVectorXd& datum,
                                            const bool add) {
  if (add) {
    data_sum += datum;
  } else {
    data_sum -= datum;
  }
}

void FAHierarchy::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
}

void FAHierarchy::set_state_from_proto(
    const google::protobuf::Message& state_) {
  auto& statecast = downcast_state(state_);
  state.mu = bayesmix::to_eigen(statecast.fa_state().mu());
  state.psi = bayesmix::to_eigen(statecast.fa_state().psi());
  state.eta = bayesmix::to_eigen(statecast.fa_state().eta());
  state.lambda = bayesmix::to_eigen(statecast.fa_state().lambda());
  state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
  compute_wood_factors(state.cov_wood, state.cov_logdet, state.lambda,
                       state.psi_inverse);
  set_card(statecast.cardinality());
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
FAHierarchy::get_state_proto() const {
  bayesmix::FAState state_;
  bayesmix::to_proto(state.mu, state_.mutable_mu());
  bayesmix::to_proto(state.psi, state_.mutable_psi());
  bayesmix::to_proto(state.eta, state_.mutable_eta());
  bayesmix::to_proto(state.lambda, state_.mutable_lambda());

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_fa_state()->CopyFrom(state_);
  return out;
}

void FAHierarchy::set_hypers_from_proto(
    const google::protobuf::Message& hypers_) {
  auto& hyperscast = downcast_hypers(hypers_).fa_state();
  hypers->mutilde = bayesmix::to_eigen(hyperscast.mutilde());
  hypers->alpha0 = hyperscast.alpha0();
  hypers->beta = bayesmix::to_eigen(hyperscast.beta());
  hypers->phi = hyperscast.phi();
  hypers->q = hyperscast.q();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
FAHierarchy::get_hypers_proto() const {
  bayesmix::FAPriorDistribution hypers_;
  bayesmix::to_proto(hypers->mutilde, hypers_.mutable_mutilde());
  bayesmix::to_proto(hypers->beta, hypers_.mutable_beta());
  hypers_.set_alpha0(hypers->alpha0);
  hypers_.set_phi(hypers->phi);
  hypers_.set_q(hypers->q);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_fa_state()->CopyFrom(hypers_);
  return out;
}

void FAHierarchy::sample_full_cond(const bool update_params /*= false*/) {
  if (this->card == 0) {
    // No posterior update possible
    sample_prior();
  } else {
    sample_eta();
    sample_mu();
    sample_psi();
    sample_lambda();
  }
}

void FAHierarchy::sample_eta() {
  auto& rng = bayesmix::Rng::Instance().get();
  auto sigma_eta_inv_llt =
      (Eigen::MatrixXd::Identity(hypers->q, hypers->q) +
       state.lambda.transpose() * state.psi_inverse * state.lambda)
          .llt();
  if (state.eta.rows() != card) {
    state.eta = Eigen::MatrixXd::Zero(card, state.eta.cols());
  }
  Eigen::MatrixXd temp_product(
      sigma_eta_inv_llt.solve(state.lambda.transpose() * state.psi_inverse));
  auto iterator = cluster_data_idx.begin();
  for (size_t i = 0; i < card; i++, iterator++) {
    Eigen::VectorXd tempvector(dataset_ptr->row(
        *iterator));  // TODO use slicing when Eigen is updated to v3.4
    state.eta.row(i) = (bayesmix::multi_normal_prec_chol_rng(
        temp_product * (tempvector - state.mu), sigma_eta_inv_llt, rng));
  }
}

void FAHierarchy::sample_mu() {
  auto& rng = bayesmix::Rng::Instance().get();
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> sigma_mu;

  sigma_mu.diagonal() =
      (card * state.psi_inverse.diagonal().array() + hypers->phi)
          .cwiseInverse();

  Eigen::VectorXd sum = (state.eta.colwise().sum());

  Eigen::VectorXd mumean =
      sigma_mu * (hypers->phi * hypers->mutilde +
                  state.psi_inverse * (data_sum - state.lambda * sum));

  state.mu = bayesmix::multi_normal_diag_rng(mumean, sigma_mu, rng);
}

void FAHierarchy::sample_lambda() {
  auto& rng = bayesmix::Rng::Instance().get();

  Eigen::MatrixXd temp_etateta(state.eta.transpose() * state.eta);

  for (size_t j = 0; j < dim; j++) {
    auto sigma_lambda_inv_llt =
        (Eigen::MatrixXd::Identity(hypers->q, hypers->q) +
         temp_etateta / state.psi[j])
            .llt();
    Eigen::VectorXd tempsum(card);
    const Eigen::VectorXd& data_col = dataset_ptr->col(j);
    auto iterator = cluster_data_idx.begin();
    for (size_t i = 0; i < card; i++, iterator++) {
      tempsum[i] = data_col(
          *iterator);  // TODO use slicing when Eigen is updated to v3.4
    }
    tempsum = tempsum.array() - state.mu[j];
    tempsum = tempsum.array() / state.psi[j];

    state.lambda.row(j) = bayesmix::multi_normal_prec_chol_rng(
        sigma_lambda_inv_llt.solve(state.eta.transpose() * tempsum),
        sigma_lambda_inv_llt, rng);
  }
}

void FAHierarchy::sample_psi() {
  auto& rng = bayesmix::Rng::Instance().get();

  for (size_t j = 0; j < dim; j++) {
    double sum = 0;
    auto iterator = cluster_data_idx.begin();
    for (size_t i = 0; i < card; i++, iterator++) {
      sum += std::pow(
          ((*dataset_ptr)(*iterator, j) -
           state.mu[j] -  // TODO use slicing when Eigen is updated to v3.4
           state.lambda.row(j).dot(state.eta.row(i))),
          2);
    }
    state.psi[j] = stan::math::inv_gamma_rng(hypers->alpha0 + card / 2,
                                             hypers->beta[j] + sum / 2, rng);
  }
  state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
  compute_wood_factors(state.cov_wood, state.cov_logdet, state.lambda,
                       state.psi_inverse);
}

void FAHierarchy::compute_wood_factors(
    Eigen::MatrixXd& cov_wood, double& cov_logdet,
    const Eigen::MatrixXd& lambda,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& psi_inverse) {
  auto [cov_wood_, cov_logdet_] =
      bayesmix::compute_wood_chol_and_logdet(psi_inverse, lambda);
  cov_logdet = cov_logdet_;
  cov_wood = cov_wood_;
}
