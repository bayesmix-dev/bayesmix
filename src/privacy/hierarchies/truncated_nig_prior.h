#ifndef BAYESMIX_PRIVACY_HIERARCHIES_TRUNCATED_NIG_PRIOR_MODEL_H_
#define BAYESMIX_PRIVACY_HIERARCHIES_TRUNCATED_NIG_PRIOR_MODEL_H_

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "hierarchy_prior.pb.h"
#include "src/hierarchies/priors/base_prior_model.h"
#include "src/hierarchies/priors/hyperparams.h"
#include "src/utils/rng.h"

/**
 * A conjugate prior model for the univariate normal likelihood, that is
 *
 * \f[
 *      \mu \mid \sigma^2 &\sim N(\mu_0, \sigma^2 / \lambda) \\
 *      \sigma^2 &\sim InvGamma(a,b) I[\sigma^2 \in (lower, upper)]
 * \f]
 */

class TruncatedNIGPriorModel
    : public BasePriorModel<TruncatedNIGPriorModel, State::UniLS,
                            Hyperparams::NIG, bayesmix::NNIGPrior> {
 public:
  using AbstractPriorModel::ProtoHypers;
  using AbstractPriorModel::ProtoHypersPtr;

  TruncatedNIGPriorModel() = default;
  ~TruncatedNIGPriorModel() = default;

  TruncatedNIGPriorModel(double var_l, double var_u)
      : var_l(var_l), var_u(var_u) {}

  void set_var_bounds(double l, double u) {
    var_l = l;
    var_u = u;
  }

  double lpdf(const google::protobuf::Message &state_) override;

  // HACK FUNCTION to get easy access to the lpdf, needed to compute the
  // marginal without messing up stuff
  double lpdf(double mean, double var, double mu0, double lam, double a,
              double b);

  template <typename T>
  T lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    Eigen::Matrix<T, Eigen::Dynamic, 1> constrained_params =
        State::uni_ls_to_constrained(unconstrained_params);
    T log_det_jac = State::uni_ls_log_det_jac(constrained_params);
    T mean = constrained_params(0);
    T var = constrained_params(1);
    T lpdf = stan::math::normal_lpdf(mean, hypers->mean,
                                     sqrt(var / hypers->var_scaling)) +
             stan::math::inv_gamma_lpdf(var, hypers->shape, hypers->scale);
    T log_norm_constant = std::log(
        stan::math::inv_gamma_cdf(var_u, hypers->shape, hypers->scale) -
        stan::math::inv_gamma_cdf(var_l, hypers->shape, hypers->scale));
    lpdf -= log_norm_constant;
    return lpdf + log_det_jac;
  }

  State::UniLS sample(ProtoHypersPtr hier_hypers = nullptr) override;

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override;

  std::pair<double, double> get_var_bounds() {
    return std::make_pair(var_l, var_u);
  }

 protected:
  double var_l, var_u;
  void initialize_hypers() override;
};

#endif  // BAYESMIX_HIERARCHIES_PRIORS_NIG_PRIOR_MODEL_H_
