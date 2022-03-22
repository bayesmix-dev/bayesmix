#include "laplace_likelihood.h"

double LaplaceLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::double_exponential_lpdf(
      datum(0), state.mean, stan::math::sqrt(state.var / 2.0));
}

// void LaplaceLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
//                                          bool add) {
//   if (add) {
//     sum_abs_diff_curr += std::abs(state.mean - datum(0, 0));
//     cluster_data_values.push_back(datum);
//   } else {
//     sum_abs_diff_curr -= std::abs(state.mean - datum(0, 0));
//     auto it = std::find(cluster_data_values.begin(),
//     cluster_data_values.end(),
//                         datum);
//     cluster_data_values.erase(it);
//   }
// }

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

// void LaplaceLikelihood::clear_summary_statistics() {
//   cluster_data_values.clear();
//   sum_abs_diff_curr = 0;
//   sum_abs_diff_prop = 0;
// }

// double UniNormLikelihood::cluster_lpdf_from_unconstrained(
//     Eigen::VectorXd unconstrained_params) {
//   assert(unconstrained_params.size() == 2);
//   double mean = unconstrained_params(0);
//   double var = std::exp(unconstrained_params(1));
//   double out = -(data_sum_squares - 2 * mean * data_sum + card * mean *
//   mean) /
//                (2 * var);
//   out -= card * 0.5 * std::log(stan::math::TWO_PI * var);
//   return out;
// }
