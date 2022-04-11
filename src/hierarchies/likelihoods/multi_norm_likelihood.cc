#include "multi_norm_likelihood.h"

#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

double MultiNormLikelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum) const {
  return bayesmix::multi_normal_prec_lpdf(datum, state.mean, state.prec_chol,
                                          state.prec_logdet);
}

void MultiNormLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                           bool add) {
  // Check if dim is not defined yet (usually not happens if hierarchy is
  // initialized)
  if (!dim) set_dim(datum.size());
  // Updates
  if (add) {
    data_sum += datum.transpose();
    data_sum_squares += datum.transpose() * datum;
  } else {
    data_sum -= datum.transpose();
    data_sum_squares -= datum.transpose() * datum;
  }
}

void MultiNormLikelihood::set_state_from_proto(
    const google::protobuf::Message &state_, bool update_card) {
  auto &statecast = downcast_state(state_);
  state.mean = to_eigen(statecast.multi_ls_state().mean());
  state.prec = to_eigen(statecast.multi_ls_state().prec());
  state.prec_chol = to_eigen(statecast.multi_ls_state().prec_chol());
  Eigen::VectorXd diag = state.prec_chol.diagonal();
  state.prec_logdet = 2 * log(diag.array()).sum();
  if (update_card) set_card(statecast.cardinality());
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
MultiNormLikelihood::get_state_proto() const {
  bayesmix::MultiLSState state_;
  bayesmix::to_proto(state.mean, state_.mutable_mean());
  bayesmix::to_proto(state.prec, state_.mutable_prec());
  bayesmix::to_proto(state.prec_chol, state_.mutable_prec_chol());
  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_multi_ls_state()->CopyFrom(state_);
  return out;
}

void MultiNormLikelihood::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
  data_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}
