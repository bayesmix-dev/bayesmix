#ifndef BAYESMIX_HIERARCHIES_GAMMA_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_GAMMA_PRIOR_MODEL_H_

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "src/hierarchies/priors/base_prior_model.h"
#include "src/utils/rng.h"

namespace Hyperparams {
struct Gamma {
  double rate_alpha, rate_beta;
};
}  // namespace Hyperparams

class GammaPriorModel
    : public BasePriorModel<GammaPriorModel, Hyperparams::Gamma,
                            bayesmix::EmptyPrior> {
 public:
  using AbstractPriorModel::ProtoHypers;
  using AbstractPriorModel::ProtoHypersPtr;

  GammaPriorModel(double shape_ = -1, double rate_alpha_ = -1,
                  double rate_beta_ = -1);
  ~GammaPriorModel() = default;

  double lpdf(const google::protobuf::Message &state_) override;

  std::shared_ptr<google::protobuf::Message> sample(
      ProtoHypersPtr hier_hypers = nullptr) override;

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override {
    return;
  };

  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  ProtoHypersPtr get_hypers_proto() const override;
  double get_shape() const { return shape; };

 protected:
  double shape, rate_alpha, rate_beta;
  void initialize_hypers() override;
};

/* DEFINITIONS */
GammaPriorModel::GammaPriorModel(double shape_, double rate_alpha_,
                                 double rate_beta_)
    : shape(shape_), rate_alpha(rate_alpha_), rate_beta(rate_beta_) {
  create_empty_prior();
};

double GammaPriorModel::lpdf(const google::protobuf::Message &state_) {
  double rate = downcast_state(state_).general_state().data()[1];
  return stan::math::gamma_lpdf(rate, hypers->rate_alpha, hypers->rate_beta);
}

std::shared_ptr<google::protobuf::Message> GammaPriorModel::sample(
    ProtoHypersPtr hier_hypers) {
  auto &rng = bayesmix::Rng::Instance().get();

  auto params = (hier_hypers) ? hier_hypers->fake_prior()
                              : get_hypers_proto()->fake_prior();
  double rate_alpha = params.data()[0];
  double rate_beta = params.data()[1];
  double new_rate = stan::math::gamma_rng(rate_alpha, rate_beta, rng);

  bayesmix::AlgorithmState::ClusterState out;
  out.mutable_general_state()->mutable_data()->Add(shape);
  out.mutable_general_state()->mutable_data()->Add(new_rate);
  return std::make_shared<bayesmix::AlgorithmState::ClusterState>(out);
}

void GammaPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_);
  hypers->rate_alpha = hyperscast.fake_prior().data()[0];
  hypers->rate_beta = hyperscast.fake_prior().data()[1];
};

GammaPriorModel::ProtoHypersPtr GammaPriorModel::get_hypers_proto() const {
  bayesmix::Vector hypers_;
  hypers_.mutable_data()->Add(hypers->rate_alpha);
  hypers_.mutable_data()->Add(hypers->rate_beta);

  ProtoHypersPtr out = std::make_shared<ProtoHypers>();
  out->mutable_fake_prior()->CopyFrom(hypers_);
  return out;
};

void GammaPriorModel::initialize_hypers() {
  hypers->rate_alpha = rate_alpha;
  hypers->rate_beta = rate_beta;

  // Checks
  if (shape <= 0) {
    throw std::runtime_error("shape must be positive");
  }
  if (rate_alpha <= 0) {
    throw std::runtime_error("rate_alpha must be positive");
  }
  if (rate_beta <= 0) {
    throw std::runtime_error("rate_beta must be positive");
  }
}

#endif  // BAYESMIX_HIERARCHIES_GAMMA_PRIOR_MODEL_H_
