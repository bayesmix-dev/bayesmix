#include "truncated_nig_prior.h"

void TruncatedNIGPriorModel::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = prior->fixed_values().mean();
    hypers->var_scaling = prior->fixed_values().var_scaling();
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    if (hypers->var_scaling <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
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

double TruncatedNIGPriorModel::lpdf(const google::protobuf::Message &state_) {
  auto &state = downcast_state(state_).uni_ls_state();
  double target =
      stan::math::normal_lpdf(state.mean(), hypers->mean,
                              sqrt(state.var() / hypers->var_scaling)) +
      stan::math::inv_gamma_lpdf(state.var(), hypers->shape, hypers->scale);
  double log_norm_constant =
      std::log(stan::math::inv_gamma_cdf(var_u, hypers->shape, hypers->scale) -
               stan::math::inv_gamma_cdf(var_l, hypers->shape, hypers->scale));
  return target - log_norm_constant;
}

double TruncatedNIGPriorModel::lpdf(double mean, double var, double mu0,
                                    double lam, double a, double b) {
  double target = stan::math::normal_lpdf(mean, mu0, sqrt(var / lam)) +
                  stan::math::inv_gamma_lpdf(var, a, b);
  double log_norm_constant = std::log(stan::math::inv_gamma_cdf(var_u, a, b) -
                                      stan::math::inv_gamma_cdf(var_l, a, b));
  return target - log_norm_constant;
}

State::UniLS TruncatedNIGPriorModel::sample(ProtoHypersPtr hier_hypers) {
  auto &rng = bayesmix::Rng::Instance().get();
  auto params = (hier_hypers) ? hier_hypers->nnig_state()
                              : get_hypers_proto()->nnig_state();

  State::UniLS out;
  // sample the variance via the inverse-cdf method
  out.var = bayesmix::sample_truncated_inv_gamma(
      params.shape(), params.scale(), var_l, var_u, rng);
  out.mean = stan::math::normal_rng(params.mean(),
                                    sqrt(out.var / params.var_scaling()), rng);
  return out;
}

void TruncatedNIGPriorModel::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();

  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void TruncatedNIGPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).nnig_state();
  hypers->mean = hyperscast.mean();
  hypers->var_scaling = hyperscast.var_scaling();
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
TruncatedNIGPriorModel::get_hypers_proto() const {
  bayesmix::NIGDistribution hypers_;
  hypers_.set_mean(hypers->mean);
  hypers_.set_var_scaling(hypers->var_scaling);
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_nnig_state()->CopyFrom(hypers_);
  return out;
}
