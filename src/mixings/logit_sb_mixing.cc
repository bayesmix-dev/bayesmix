#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>

#include "dirichlet_mixing.h"
#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"

void LogitSBMixing::initialize(const unsigned int n_clust /*= 1*/) {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  num_clusters = n_clust;
  initialize_state();
}

void LogitSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_values()) {
    Eigen::VectorXd prior_vec =
        bayesmix::to_eigen(priorcast->fixed_values().coefficients());
    dim = state.regression_coeffs.size();
    state.regression_coeffs = Eigen::MatrixXd(dim, num_clusters);
    for (int i = 0; i < num_clusters; i++) {
      state.regression_coeffs << prior_vec;
    }

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
  auto &statecast =
      google::protobuf::internal::down_cast<const bayesmix::LogSBState &>(
          state_);
  state.regression_coeffs = bayesmix::to_eigen(statecast.regression_coeffs());
}

void LogitSBMixing::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::LogSBState state_;
  bayesmix::to_proto(state.regression_coeffs,
                     state_.mutable_regression_coeffs());
  google::protobuf::internal::down_cast<bayesmix::LogSBState *>(out)->CopyFrom(
      state_);
}

Eigen::VectorXd get_weights(const Eigen::VectorXd &covariate) const {
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
