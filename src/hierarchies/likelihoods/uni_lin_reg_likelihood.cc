#include "uni_lin_reg_likelihood.h"

#include "src/utils/eigen_utils.h"

void UniLinRegLikelihood::set_state_from_proto(
    const google::protobuf::Message &state_, bool update_card) {
  auto &statecast = downcast_state(state_);
  state.regression_coeffs =
      bayesmix::to_eigen(statecast.lin_reg_uni_ls_state().regression_coeffs());
  state.var = statecast.lin_reg_uni_ls_state().var();
  if (update_card) set_card(statecast.cardinality());
}

void UniLinRegLikelihood::clear_summary_statistics() {
  mixed_prod = Eigen::VectorXd::Zero(dim);
  data_sum_squares = 0.0;
  covar_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
UniLinRegLikelihood::get_state_proto() const {
  bayesmix::LinRegUniLSState state_;
  bayesmix::to_proto(state.regression_coeffs,
                     state_.mutable_regression_coeffs());
  state_.set_var(state.var);
  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_lin_reg_uni_ls_state()->CopyFrom(state_);
  return out;
}

double UniLinRegLikelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) const {
  return stan::math::normal_lpdf(
      datum(0), state.regression_coeffs.dot(covariate), sqrt(state.var));
}

void UniLinRegLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                           const Eigen::RowVectorXd &covariate,
                                           bool add) {
  if (add) {
    data_sum_squares += datum(0) * datum(0);
    covar_sum_squares += covariate.transpose() * covariate;
    mixed_prod += datum(0) * covariate.transpose();
  } else {
    data_sum_squares -= datum(0) * datum(0);
    covar_sum_squares -= covariate.transpose() * covariate;
    mixed_prod -= datum(0) * covariate.transpose();
  }
}
