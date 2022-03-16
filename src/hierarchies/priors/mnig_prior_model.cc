#include "mnig_prior_model.h"

double MNIGPriorModel::lpdf(const google::protobuf::Message &state_) {
  auto &state = downcast_state(state_).lin_reg_uni_ls_state();
  Eigen::VectorXd regression_coeffs =
      bayesmix::to_eigen(state.regression_coeffs());
  double target = stan::math::multi_normal_prec_lpdf(
      regression_coeffs, hypers->mean, hypers->var_scaling / state.var());
  target +=
      stan::math::inv_gamma_lpdf(state.var(), hypers->shape, hypers->scale);
  return target;
}

std::shared_ptr<google::protobuf::Message> MNIGPriorModel::sample(
    bool use_post_hypers) {
  auto &rng = bayesmix::Rng::Instance().get();
  Hyperparams::MNIG params = use_post_hypers ? post_hypers : *hypers;

  double var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  Eigen::VectorXd regression_coeffs = stan::math::multi_normal_prec_rng(
      params.mean, params.var_scaling / var, rng);

  bayesmix::AlgorithmState::ClusterState state;
  // bayesmix::Vector regression_coeffs_proto;
  bayesmix::to_proto(
      regression_coeffs,
      state.mutable_lin_reg_uni_ls_state()->mutable_regression_coeffs());
  // state.mutable_lin_reg_uni_ls_state()->mutable_regression_coeffs()->CopyFrom(regression_coeffs_proto);
  state.mutable_lin_reg_uni_ls_state()->set_var(var);

  return std::make_shared<bayesmix::AlgorithmState::ClusterState>(state);
}

void MNIGPriorModel::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void MNIGPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).lin_reg_uni_state();
  hypers->mean = bayesmix::to_eigen(hyperscast.mean());
  hypers->var_scaling = bayesmix::to_eigen(hyperscast.var_scaling());
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
MNIGPriorModel::get_hypers_proto() const {
  bayesmix::MultiNormalIGDistribution hypers_;
  bayesmix::to_proto(hypers->mean, hypers_.mutable_mean());
  bayesmix::to_proto(hypers->var_scaling, hypers_.mutable_var_scaling());
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_lin_reg_uni_state()->CopyFrom(hypers_);
  return out;
}

void MNIGPriorModel::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = bayesmix::to_eigen(prior->fixed_values().mean());
    dim = hypers->mean.size();
    hypers->var_scaling =
        bayesmix::to_eigen(prior->fixed_values().var_scaling());
    hypers->var_scaling_inv = stan::math::inverse_spd(hypers->var_scaling);
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    if (dim != hypers->var_scaling.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    bayesmix::check_spd(hypers->var_scaling);
    if (hypers->shape <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (hypers->scale <= 0) {
      throw std::invalid_argument("scale parameter must be > 0");
    }
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}
