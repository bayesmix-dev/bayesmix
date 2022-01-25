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

double UniNormLikelihood::cluster_lpdf_from_unconstrained(
    Eigen::VectorXd unconstrained_params) {
  assert(unconstrained_params.size() == 2);
  double mean = unconstrained_params(0);
  double var = std::exp(unconstrained_params(1));
  double out = -(data_sum_squares - 2 * mean * data_sum + card * mean * mean) /
               (2 * var);
  out -= card * 0.5 * std::log(stan::math::TWO_PI * var);
  return out;
}
