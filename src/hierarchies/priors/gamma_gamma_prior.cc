#include "gamma_gamma_prior.h"

double GGPriorModel::lpdf(const google::protobuf::Message &state_) {
  // Downcast state
  auto &state = downcast_state(state_).sr_state();
  double target = 0.;
  target +=
      stan::math::gamma_lpdf(state.shape(), hypers->a_shape, hypers->a_rate);
  target +=
      stan::math::gamma_lpdf(state.rate(), hypers->b_shape, hypers->b_rate);
  return target;
}

State::ShapeRate GGPriorModel::sample(ProtoHypersPtr hier_hypers) {
  // Random seed
  auto &rng = bayesmix::Rng::Instance().get();

  // Get params to use
  auto params = get_hypers_proto()->gg_state();
  State::ShapeRate out;
  out.shape = stan::math::gamma_rng(params.a_shape(), params.a_rate(), rng);
  out.rate = stan::math::gamma_rng(params.b_shape(), params.b_rate(), rng);
  return out;
}

void GGPriorModel::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void GGPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).gg_state();
  hypers->a_shape = hyperscast.a_shape();
  hypers->a_rate = hyperscast.a_rate();
  hypers->b_shape = hyperscast.b_shape();
  hypers->b_rate = hyperscast.b_rate();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
GGPriorModel::get_hypers_proto() const {
  bayesmix::GamGamDistribution hypers_;
  hypers_.set_a_shape(hypers->a_shape);
  hypers_.set_a_rate(hypers->a_rate);
  hypers_.set_b_shape(hypers->b_shape);
  hypers_.set_b_rate(hypers->b_rate);
  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_gg_state()->CopyFrom(hypers_);
  return out;
}

void GGPriorModel::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->a_shape = prior->fixed_values().a_shape();
    hypers->a_rate = prior->fixed_values().a_rate();
    hypers->b_shape = prior->fixed_values().b_shape();
    hypers->b_rate = prior->fixed_values().b_rate();
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}
