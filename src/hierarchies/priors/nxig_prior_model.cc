#include "nxig_prior_model.h"

void NxIGPriorModel::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = prior->fixed_values().mean();
    hypers->var = prior->fixed_values().var();
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    if (hypers->var <= 0) {
      throw std::invalid_argument("Variance parameter must be > 0");
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

double NxIGPriorModel::lpdf(const google::protobuf::Message &state_) {
  auto &state = downcast_state(state_).uni_ls_state();
  double target =
      stan::math::normal_lpdf(state.mean(), hypers->mean, sqrt(hypers->var)) +
      stan::math::inv_gamma_lpdf(state.var(), hypers->shape, hypers->scale);
  return target;
}

std::shared_ptr<google::protobuf::Message> NxIGPriorModel::sample(
    bayesmix::AlgorithmState::HierarchyHypers hier_hypers) {
  auto &rng = bayesmix::Rng::Instance().get();
  auto params = hier_hypers.nnxig_state();

  double var = stan::math::inv_gamma_rng(params.shape(), params.scale(), rng);
  double mean = stan::math::normal_rng(params.mean(), sqrt(params.var()), rng);

  bayesmix::AlgorithmState::ClusterState state;
  state.mutable_uni_ls_state()->set_mean(mean);
  state.mutable_uni_ls_state()->set_var(var);
  return std::make_shared<bayesmix::AlgorithmState::ClusterState>(state);
};

// std::shared_ptr<google::protobuf::Message> NxIGPriorModel::sample(
//     bool use_post_hypers) {
//   auto &rng = bayesmix::Rng::Instance().get();
//   Hyperparams::NxIG params = use_post_hypers ? post_hypers : *hypers;

//   double var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
//   double mean = stan::math::normal_rng(params.mean, sqrt(params.var), rng);

//   bayesmix::AlgorithmState::ClusterState state;
//   state.mutable_uni_ls_state()->set_mean(mean);
//   state.mutable_uni_ls_state()->set_var(var);
//   return std::make_shared<bayesmix::AlgorithmState::ClusterState>(state);
// };

void NxIGPriorModel::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void NxIGPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).nnxig_state();
  hypers->mean = hyperscast.mean();
  hypers->var = hyperscast.var();
  hypers->scale = hyperscast.scale();
  hypers->shape = hyperscast.shape();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
NxIGPriorModel::get_hypers_proto() const {
  bayesmix::NxIGDistribution hypers_;
  hypers_.set_mean(hypers->mean);
  hypers_.set_var(hypers->var);
  hypers_.set_shape(hypers->shape);
  hypers_.set_scale(hypers->scale);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_nnxig_state()->CopyFrom(hypers_);
  return out;
}
