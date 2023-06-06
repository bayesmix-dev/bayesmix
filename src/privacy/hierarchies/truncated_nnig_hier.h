#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_

#include "hierarchy_id.pb.h"
#include "src/hiearrchies/likelihoods/uni_norm_likelihood.h"
#include "src/hiearrchies/updaters/nnig_updater.h"
#include "src/hierarchies/base_hierarchy.h"
#include "truncated_nig_hier.h"

/**
 * A Normal- Normal Inverse Gamma hierarchy with inflated variance to
 * satisfy approximate local differential privacy
 *
 * This class represents a hierarchical model where data are distributed
 * according to a Normal likelihood (see the `UniNormLikelihood` class for
 * details). The likelihood parameters have a Normal-InverseGamma centering
 * distribution (see the `TruncatedNIGPriorModel` class for details),
 * such that the variance is Inverse-Gamma Distributed with support [var_l,
 * var_u].
 *
 * The state is composed of mean and variance.
 */

class PrivateNIGHier : public BaseHierarchy<NNIGHierarchy, UniNormLikelihood,
                                            TruncatedNIGPriorModel> {
 public:
  PrivateNIGHier() = default;
  ~PrivateNIGHier() = default;

  PrivateNIGHier(double var_l, double var_u = stan::math::INFTY) {
    this->prior = std::make_shared<TruncatedNIGPriorModel>(var_l, var_u);
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() { updater = std::make_shared<NNIGUpdater>(); }

  //! Initializes state parameters to appropriate values
  void initialize_state() override {
    // Get hypers
    auto hypers = prior->get_hypers();
    // Initialize likelihood state
    State::UniLS state;
    state.mean = hypers.mean;
    state.var = hypers.scale / (hypers.shape + 1);
    like->set_state(state);
  };

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param hier_params  Pointer to the container of (prior or posterior)
  //! hyperparameter values
  //! @param datum        Point which is to be evaluated
  //! @return             The evaluation of the lpdf
  double marg_lpdf(ProtoHypersPtr hier_params,
                   const Eigen::RowVectorXd &datum) const override {
    double mu0 = hier_params->nnig_state().mean();
    double lambda0 = hier_params->nnig_state().var_scaling();
    double alpha0 = hier_params->nnig_state().shape();
    double beta0 = hier_params->nnig_state().scale();
    auto prior_cast = std::dynamic_pointer_cast<TruncatedNIGPriorModel>(prior);
    auto [var_l, var_u] = prior_cast->get_var_bounds();

    double prior_lpdf =
        prior->lpdf(state.mean, state.var, mu0, lambda0, alpha0, beta0);
    double like_lpdf = like->lpdf(datum);

    double mu_n = (lambda0 * mu0 + datum(0)) / (lambda0 + 1);
    double alpha_n = alpha0 + 0.5;
    double lambda_n = lambda0 + 1;
    double beta_n = beta0 + (0.5 * lambda0 / (lambda0 + 1)) *
                                (datum(0) - mu0) * (datum(0) - mu0);
    double post_lpdf =
        prior->lpdf(state.mean, state.var, mu_n, lambda_n, alpha_n, beta_n);
    return prior_lpdf + like_lpdf - post_lpdf;
  }
};

void eval_private_nnig_lpdf(const std::shared_ptr<BaseAlgorithm> algo,
                            BaseCollector *const collector,
                            const Eigen::MatrixXd &grid, double var_l);

#endif  // BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
