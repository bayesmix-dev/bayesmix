#include "dirichlet_mixing.h"

#include <Eigen/Dense>

void LogitSBMixing::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  initialize_state();
}

void LogitSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_values()) {
    state.regression_coeffs = bayesmix::to_eigen(
                              priorcast->fixed_values().coefficients());
    dim = state.regression_coeffs.size();

  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

void LogitSBMixing::update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      unsigned int n) {
  return;  // TODO stuff with MH
}

void LogitSBMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  return;  // TODO
}
void LogitSBMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  return;  // TODO
}

Eigen::VectorXd get_weights(const Eigen::VectorXd &covariate) const {
  // TODO set n_clust aka regr_coeffs.rows() somehow!
  Eigen::VectorXd eta(n_clust);
  Eigen::VectorXd weights(n_clust);
  for (int h = 0; h < n_clust; h++) {
    eta(h) = covariate.dot(state.regression_coeffs.col(h));
    weights(h) = sigmoid(eta(h));
    for (int k = 0; k < h - 1; k++) {
      weights(h) *= sigmoid(-eta(k));
    }
  }
  return weights;
}
