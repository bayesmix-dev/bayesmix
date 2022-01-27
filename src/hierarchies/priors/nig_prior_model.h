#ifndef BAYESMIX_HIERARCHIES_NIG_PRIOR_MODEL_H_
#define BAYESMIX_HIERARCHIES_NIG_PRIOR_MODEL_H_

// #include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

// #include "algorithm_state.pb.h"
#include "base_prior_model.h"
#include "hierarchy_prior.pb.h"
#include "hyperparams.h"
#include "src/utils/rng.h"

class NIGPriorModel : public BasePriorModel<NIGPriorModel, Hyperparams::NIG,
                                            bayesmix::NNIGPrior> {
 public:
  NIGPriorModel() = default;
  ~NIGPriorModel() = default;

  double lpdf(const google::protobuf::Message &state_) override;

  template <typename T>
  T lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    Eigen::Matrix<T, Eigen::Dynamic, 1> constrained_params =
        State::uni_ls_to_constrained(unconstrained_params);
    T log_det_jac = State::uni_ls_log_det_jac(constrained_params);
    T mean = constrained_params(0);
    T var = constrained_params(1);
    T lpdf = stan::math::normal_lpdf(mean, hypers.mean,
                                     sqrt(var / hypers.var_scaling)) +
             stan::math::inv_gamma_lpdf(var, hypers.shape, hypers.scale);

    return lpdf + log_det_jac;
  }

  double lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) const override {
    return this->lpdf_from_unconstrained<double>(unconstrained_params);
  }

  std::shared_ptr<google::protobuf::Message> sample(
      bool use_post_hypers) override;

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

 protected:
  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override;

  void initialize_hypers() override;
};

#endif  // BAYESMIX_HIERARCHIES_NIG_PRIOR_MODEL_H_
