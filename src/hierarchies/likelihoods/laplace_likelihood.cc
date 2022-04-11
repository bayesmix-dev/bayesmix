#include "laplace_likelihood.h"

double LaplaceLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::double_exponential_lpdf(
      datum(0), state.mean, stan::math::sqrt(state.var / 2.0));
}

void LaplaceLikelihood::set_state_from_proto(
    const google::protobuf::Message &state_, bool update_card) {
  auto &statecast = downcast_state(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.var = statecast.uni_ls_state().var();
  if (update_card) set_card(statecast.cardinality());
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
LaplaceLikelihood::get_state_proto() const {
  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_uni_ls_state()->set_mean(state.mean);
  out->mutable_uni_ls_state()->set_var(state.var);
  return out;
}
