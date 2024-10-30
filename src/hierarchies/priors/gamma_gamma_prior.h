#ifndef BAYESMIX_HIERARCHIES_PRIORS_GG_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_PRIORS_GG_PRIOR_MODEL_H_

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_prior_model.h"
#include "hierarchy_prior.pb.h"
#include "hyperparams.h"
#include "src/utils/rng.h"

/*
 * Prior model for `ShapeRate` states.
 * This class assumes that the shape and rate are independent and given
 * Gamma-distributed priors
 */
class GGPriorModel
    : public BasePriorModel<GGPriorModel, State::ShapeRate, Hyperparams::GG,
                            bayesmix::GGPrior> {
 public:
  using AbstractPriorModel::ProtoHypers;
  using AbstractPriorModel::ProtoHypersPtr;

  GGPriorModel() = default;
  ~GGPriorModel() = default;

  double lpdf(const google::protobuf::Message &state_) override;

  template <typename T>
  T lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    Eigen::Matrix<T, Eigen::Dynamic, 1> constrained_params =
        State::shape_rate_to_constrained(unconstrained_params);
    // std::cout << "constrained_params: " << constrained_params << std::endl;
    T log_det_jac = State::shape_rate_log_det_jac(constrained_params);
    T shape = constrained_params(0);
    T rate = constrained_params(1);
    T lpdf = stan::math::gamma_lpdf(shape, hypers->a_shape, hypers->a_rate) +
             stan::math::gamma_lpdf(rate, hypers->a_shape, hypers->a_rate);

    return lpdf + log_det_jac;
  }

  State::ShapeRate sample(ProtoHypersPtr hier_hypers = nullptr) override;

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  unsigned int get_dim() const { return dim; };

  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override;

 protected:
  void initialize_hypers() override;

  unsigned int dim;
};

#endif  // BAYESMIX_HIERARCHIES_PRIORS_GG_PRIOR_MODEL_H_
