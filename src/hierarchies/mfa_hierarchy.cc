#include "mfa_hierarchy.h"

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

double MFAHierarchy::like_lpdf(const Eigen::RowVectorXd& datum) const {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  double base = 2 * (Eigen::MatrixXd(state.prec_chol.matrixL()))
                        .diagonal()
                        .array()
                        .log()
                        .sum() +
                NEG_LOG_SQRT_TWO_PI * dim;
  double exp =
      ((datum.transpose() - state.mu)
           .dot(state.prec_chol.solve((datum.transpose() - state.mu))));
  return -0.5 * (base + exp);
}

MFA::State MFAHierarchy::draw(const MFA::Hyperparams& params) {
  auto& rng = bayesmix::Rng::Instance().get();
  MFA::State out;
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
  out.prec_chol = (out.lambda * out.lambda.transpose() +
                   Eigen::MatrixXd(out.psi.asDiagonal()))
                      .llt();

  return out;
}

void MFAHierarchy::initialize_state() {
  std::cout << "init state" << std::endl;
  state.mu = hypers->mutilde;
  state.psi = hypers->beta / (hypers->alpha0 + 1.);
  state.eta = Eigen::MatrixXd::Zero(card, hypers->q);
  state.lambda = Eigen::MatrixXd::Zero(dim, hypers->q);
  state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
  state.prec_chol = (state.lambda * state.lambda.transpose() +
                     Eigen::MatrixXd(state.psi.asDiagonal()))
                        .llt();
}

void MFAHierarchy::initialize_hypers() {
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
      std::cout << "No mutilde found. Initializing with mean." << std::endl;
      hypers->mutilde = dataset_ptr->colwise().mean();
      dim = hypers->mutilde.size();
    }
    if (hypers->beta.size() == 0) {
      std::cout << "No beta found. Initializing with scaled precision matrix "
                   "diagonal."
                << std::endl;
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
    if (hypers->phi <= 0) {  // TODO check
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
  } else {
    data_sum -= datum;
  }
}

void MFAHierarchy::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
}

void MFAHierarchy::set_state_from_proto(
    const google::protobuf::Message& state_) {
  auto& statecast = downcast_state(state_);
  state.mu = bayesmix::to_eigen(statecast.mfa_state().mu());
  state.psi = bayesmix::to_eigen(statecast.mfa_state().psi());
  state.eta = bayesmix::to_eigen(statecast.mfa_state().eta());
  state.lambda = bayesmix::to_eigen(statecast.mfa_state().lambda());
  state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
  state.prec_chol = (state.lambda * state.lambda.transpose() +
                     Eigen::MatrixXd(state.psi.asDiagonal()))
                        .llt();
  set_card(statecast.cardinality());
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
MFAHierarchy::get_state_proto() const {
  bayesmix::MFAState state_;
  bayesmix::to_proto(state.mu, state_.mutable_mu());
  bayesmix::to_proto(state.psi, state_.mutable_psi());
  bayesmix::to_proto(state.eta, state_.mutable_eta());
  bayesmix::to_proto(state.lambda, state_.mutable_lambda());

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_mfa_state()->CopyFrom(state_);
  return out;
}

void MFAHierarchy::set_hypers_from_proto(
    const google::protobuf::Message& hypers_) {
  auto& hyperscast = downcast_hypers(hypers_).mfa_state();
  hypers->mutilde = bayesmix::to_eigen(hyperscast.mutilde());
  hypers->alpha0 = hyperscast.alpha0();
  hypers->beta = bayesmix::to_eigen(hyperscast.beta());
  hypers->phi = hyperscast.phi();
  hypers->q = hyperscast.q();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
MFAHierarchy::get_hypers_proto() const {
  bayesmix::MFAPriorDistribution hypers_;
  bayesmix::to_proto(hypers->mutilde, hypers_.mutable_mutilde());
  bayesmix::to_proto(hypers->beta, hypers_.mutable_beta());
  hypers_.set_alpha0(hypers->alpha0);
  hypers_.set_phi(hypers->phi);
  hypers_.set_q(hypers->q);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_mfa_state()->CopyFrom(hypers_);
  return out;
}

void MFAHierarchy::sample_full_cond(bool update_params) {
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

void MFAHierarchy::sample_eta() {
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

void MFAHierarchy::sample_mu() {
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

void MFAHierarchy::sample_lambda() {
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

void MFAHierarchy::sample_psi() {
  auto& rng = bayesmix::Rng::Instance().get();
  /*//(LAMBDA*ETA^T)^T = ETA*LAMBDA^T  (dim*q q*card)^T
  //(data.rowwise()-mu-LAMBDA*ETA^T).square().colwise().sum()
  Eigen::MatrixXd lambda_eta(state.eta * state.lambda.transpose());
  Eigen::MatrixXd tempdata(card, dim);
  auto iterator = cluster_data_idx.begin();
  for (size_t i = 0; i < card;
       i++, iterator++) {  // TODO use slicing when Eigen is updated to v3.4
    tempdata.row(i) = dataset_ptr->row(*iterator);
  }


  Eigen::VectorXd sum =
      (((tempdata - lambda_eta).rowwise() -
  state.mu.transpose()).array().square()) .colwise() .sum();
  std::cout<<sum<<std::endl;

  for (size_t j = 0; j < dim; j++) {
    state.psi[j] = stan::math::inv_gamma_rng(
        hypers->alpha0 + card / 2, hypers->beta[j] + sum[j] / 2, rng);
  }*/

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
  state.prec_chol = (state.lambda * state.lambda.transpose() +
                     Eigen::MatrixXd(state.psi.asDiagonal()))
                        .llt();
}
