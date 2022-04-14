#include "uni_norm_likelihood.h"

double UniNormLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::normal_lpdf(datum(0), state.mean, sqrt(state.var));
}

void UniNormLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                         bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
}

void UniNormLikelihood::set_state_from_proto(
    const google::protobuf::Message &state_, bool update_card) {
  auto &statecast = downcast_state(state_);
  state.mean = statecast.uni_ls_state().mean();
  state.var = statecast.uni_ls_state().var();
  if (update_card) set_card(statecast.cardinality());
}

void UniNormLikelihood::set_state(
    const States::UniLS &state_, bool update_card) {
  
  int old_card;
  if (! update_card) {
    old_card = state.card;
  }
  state = state_;
  if (! update_card) {
    state.card = old_card;
  }
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
UniNormLikelihood::get_state_proto() const {
  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_uni_ls_state()->set_mean(state.mean);
  out->mutable_uni_ls_state()->set_var(state.var);
  return out;
}

void UniNormLikelihood::clear_summary_statistics() {
  data_sum = 0;
  data_sum_squares = 0;
}
