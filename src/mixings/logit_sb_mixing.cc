#include "logit_sb_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

void LogitSBMixing::initialize(const unsigned int n_clust /*= 1*/) {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  num_clusters = n_clust;
  auto priorcast = cast_prior();
  precision = stan::math::inverse_spd(
    bayesmix::to_eigen(priorcast->normal_prior().var()));
  initialize_state();
}

void LogitSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_normal_prior()) {
    Eigen::VectorXd prior_vec =
        bayesmix::to_eigen(priorcast->normal_prior().mean());
    dim = prior_vec.size();
    if (dim != precision.cols()) {
      throw std::invalid_argument(
        "Hyperparameters dimensions are not consisent");
    }
    if (priorcast->proposal_var() <= 0) {
      throw std::invalid_argument("Proposal variance parameter must be > 0");
    }
    if (priorcast->step_size() <= 0) {
      throw std::invalid_argument("Step size parameter must be > 0");
    }
    state.regression_coeffs = Eigen::MatrixXd(dim, num_clusters);
    for (int i = 0; i < num_clusters; i++) {
      state.regression_coeffs.col(i) = prior_vec;
    }

  } else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

void LogitSBMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations,
    const unsigned int n) {
  // Langevin-Adjusted Metropolis-Hastings step
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd state_c = state.regression_coeffs;
  auto priorcast = cast_prior();
  Eigen::VectorXd prior_mean = bayesmix::to_eigen(
      priorcast->normal_prior().mean());
  double prop_var = priorcast->proposal_var();
  double step = priorcast->step_size();
  // Loop over clusters
  for (int h = 0; h < unique_values.size(); h++) {
    // ...
    Eigen::VectorXd prop_mean;  // TODO
    Eigen::VectorXd state_prop;  // TODO
    double prior_ratio = -0.5 * (
      (state_prop - prior_mean).transpose() * precision * (state_prop - prior_mean) -
      (state_c - prior_mean).transpose() * precision * (state_c - prior_mean)
    );
    double like_ratio;  // TODO
    double prop_ratio = (-0.5 / prop_var) * (
      (state_prop - prop_mean).dot(state_prop - prop_mean) -
      (state_c - prop_mean).dot(state_c - prop_mean)
    );
    double accept_ratio = prior_ratio + like_ratio - prop_ratio;
    // Accept with probability ratio
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p < std::exp(logratio)) {
      state.regression_coeffs = state_prop;
    }
  }
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

Eigen::VectorXd LogitSBMixing::get_weights(
    const Eigen::VectorXd &covariate) const {
  Eigen::VectorXd eta(num_clusters);
  Eigen::VectorXd weights(num_clusters);
  for (int h = 0; h < num_clusters; h++) {
    eta(h) = covariate.dot(state.regression_coeffs.col(h));
    weights(h) = sigmoid(eta(h));
    for (int k = 0; k < h - 1; k++) {
      weights(h) *= sigmoid(-eta(k));
    }
  }
  return weights;
}
