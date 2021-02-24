#include "logit_sb_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

void LogitSBMixing::initialize() {
  if (prior == nullptr) {
    throw std::invalid_argument("Mixing prior was not provided");
  }
  auto priorcast = cast_prior();
  num_clusters = priorcast->num_clusters();
  initialize_state();
}

void LogitSBMixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_normal_prior()) {
    Eigen::VectorXd prior_vec =
        bayesmix::to_eigen(priorcast->normal_prior().mean());
    dim = prior_vec.size();
    state.precision = stan::math::inverse_spd(
        bayesmix::to_eigen(priorcast->normal_prior().var()));
    if (dim != state.precision.cols()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consisent");
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

Eigen::VectorXd LogitSBMixing::grad_log_full_cond(
    const Eigen::VectorXd &alpha, const std::vector<bool> &is_curr_clus,
    const std::vector<bool> &is_subseq_clus) {
  auto priorcast = cast_prior();
  Eigen::VectorXd prior_mean =
      bayesmix::to_eigen(priorcast->normal_prior().mean());
  Eigen::VectorXd grad = state.precision * (prior_mean - alpha);
  for (int i = 0; i < is_curr_clus.size(); i++) {
    double sig = sigmoid(covariates_ptr->row(i).dot(alpha));
    double coeff = double(is_curr_clus[i]) * (1.0 - sig) -
                   double(is_subseq_clus[i]) * sig;
    grad += coeff * covariates_ptr->row(i);
  }
  return grad;
}

double LogitSBMixing::log_like(const Eigen::VectorXd &alpha,
                               const std::vector<bool> &is_curr_clus,
                               const std::vector<bool> &is_subseq_clus) {
  double like = 0.0;
  for (int i = 0; i < is_curr_clus.size(); i++) {
    double sig = sigmoid(covariates_ptr->row(i).dot(alpha));
    like += double(is_curr_clus[i]) * std::log(sig) +
            double(is_subseq_clus[i]) * std::log(1.0 - sig);
  }
  return like;
}

void LogitSBMixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  // Langevin-Adjusted Metropolis-Hastings step
  unsigned int n = allocations.size();
  auto &rng = bayesmix::Rng::Instance().get();
  auto priorcast = cast_prior();
  Eigen::VectorXd prior_mean =
      bayesmix::to_eigen(priorcast->normal_prior().mean());
  double step = priorcast->step_size();
  double prop_var = std::sqrt(2.0 * step);
  // Loop over clusters
  for (int h = 0; h < unique_values.size(); h++) {
    Eigen::VectorXd state_c = state.regression_coeffs.col(h);
    // Compute allocation indicators
    std::vector<bool> is_curr_clus(n, false);
    std::vector<bool> is_subseq_clus(n, false);
    for (int i = 0; i < n; i++) {
      if (allocations[i] == h) {
        is_curr_clus[i] = true;
      } else if (allocations[i] > h) {
        is_subseq_clus[i] = true;
      }
    }
    // Draw proposed state from its distribution
    Eigen::VectorXd prop_mean =
        state_c +
        step * grad_log_full_cond(state_c, is_curr_clus, is_subseq_clus);
    auto prop_covar = prop_var * Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd state_prop =
        stan::math::multi_normal_rng(prop_mean, prop_covar, rng);
    // Compute acceptance ratio
    double prior_ratio =
        -0.5 * ((state_prop - prior_mean).transpose() * state.precision *
                    (state_prop - prior_mean) -
                (state_c - prior_mean).transpose() * state.precision *
                    (state_c - prior_mean))(0);
    double like_ratio = log_like(state_prop, is_curr_clus, is_subseq_clus) -
                        log_like(state_c, is_curr_clus, is_subseq_clus);
    double prop_ratio = (-0.5 / prop_var) *
                        ((state_prop - prop_mean).dot(state_prop - prop_mean) -
                         (state_c - prop_mean).dot(state_c - prop_mean));
    double log_accept_ratio = prior_ratio + like_ratio - prop_ratio;
    // Accept with probability ratio
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p < std::exp(log_accept_ratio)) {
      state.regression_coeffs.col(h) = state_prop;
      // TODO should we set an internal flag proposal_was_accepted?
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
    const Eigen::VectorXd &covariate /*= Eigen::VectorXd(0)*/) const {
  // TODO design choice: no check on covariate
  // Compute eta
  std::vector<double> eta(num_clusters);
  for (int h = 0; h < num_clusters; h++) {
    eta[h] = covariate.dot(state.regression_coeffs.col(h));
  }
  // Compute cumulative products
  std::vector<double> cumprod(num_clusters + 1, 1.0);
  for (int h = 1; h < num_clusters + 1; h++) {
    cumprod[h] = cumprod[h - 1] * sigmoid(-eta[h - 1]);
  }
  // Compute weights
  Eigen::VectorXd weights(num_clusters);
  for (int h = 0; h < num_clusters; h++) {
    weights(h) = sigmoid(eta[h]) * cumprod[h];
  }
  return weights;
}
