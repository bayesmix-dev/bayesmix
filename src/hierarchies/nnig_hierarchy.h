#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/uni_norm_likelihood.h"
#include "priors/nig_prior_model.h"
#include "updaters/nnig_updater.h"

/**
 * Conjugate Normal Normal-InverseGamma hierarchy for univariate data.
 *
 * This class represents a hierarchical model where data are distributed
 * according to a Normal likelihood (see the `UniNormLikelihood` class for
 * details). The likelihood parameters have a Normal-InverseGamma centering
 * distribution (see the `NIGPriorModel` class for details). That is:
 *
 * \f[
 *    f(x_i \mid \mu, \sigma^2) &= N(\mu,\sigma^2) \\
 *    (\mu,\sigma^2) & \sim NIG(\mu_0, \lambda_0, \alpha_0, \beta_0)
 * \f]
 *
 * The state is composed of mean and variance. The state hyperparameters are
 * \f$(\mu_0, \lambda_0, \alpha_0, \beta_0)\f$, all scalar values. Note that
 * this hierarchy is conjugate, thus the marginal distribution is available in
 * closed form
 */

class NNIGHierarchy
    : public BaseHierarchy<NNIGHierarchy, UniNormLikelihood, NIGPriorModel> {
 public:
  NNIGHierarchy() = default;
  ~NNIGHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNIG;
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
    auto params = hier_params->nnig_state();
    double sig_n = sqrt(params.scale() * (params.var_scaling() + 1) /
                        (params.shape() * params.var_scaling()));
    return stan::math::student_t_lpdf(datum(0), 2 * params.shape(),
                                      params.mean(), sig_n);
  }
};

#endif  // BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
