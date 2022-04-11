#include "fa_likelihood.h"

#include "src/utils/distributions.h"

void FALikelihood::set_state_from_proto(
    const google::protobuf::Message& state_, bool update_card) {
  auto& statecast = downcast_state(state_);
  state.mu = bayesmix::to_eigen(statecast.fa_state().mu());
  state.psi = bayesmix::to_eigen(statecast.fa_state().psi());
  state.eta = bayesmix::to_eigen(statecast.fa_state().eta());
  state.lambda = bayesmix::to_eigen(statecast.fa_state().lambda());
  state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
  compute_wood_factors(state.cov_wood, state.cov_logdet, state.lambda,
                       state.psi_inverse);
  if (update_card) set_card(statecast.cardinality());
}

void FALikelihood::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
FALikelihood::get_state_proto() const {
  bayesmix::FAState state_;
  bayesmix::to_proto(state.mu, state_.mutable_mu());
  bayesmix::to_proto(state.psi, state_.mutable_psi());
  bayesmix::to_proto(state.eta, state_.mutable_eta());
  bayesmix::to_proto(state.lambda, state_.mutable_lambda());

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_fa_state()->CopyFrom(state_);
  return out;
}

double FALikelihood::compute_lpdf(const Eigen::RowVectorXd& datum) const {
  return bayesmix::multi_normal_lpdf_woodbury_chol(
      datum, state.mu, state.psi_inverse, state.cov_wood, state.cov_logdet);
}

void FALikelihood::update_sum_stats(const Eigen::RowVectorXd& datum,
                                    bool add) {
  if (add) {
    data_sum += datum;
  } else {
    data_sum -= datum;
  }
}

void FALikelihood::compute_wood_factors(
    Eigen::MatrixXd& cov_wood, double& cov_logdet,
    const Eigen::MatrixXd& lambda,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& psi_inverse) {
  auto [cov_wood_, cov_logdet_] =
      bayesmix::compute_wood_chol_and_logdet(psi_inverse, lambda);
  cov_logdet = cov_logdet_;
  cov_wood = cov_wood_;
}
