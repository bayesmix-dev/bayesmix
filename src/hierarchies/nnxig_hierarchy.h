#ifndef BAYESMIX_HIERARCHIES_NNXIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNXIG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/uni_norm_likelihood.h"
#include "priors/nxig_prior_model.h"
#include "updaters/nnxig_updater.h"

/**
 * Semi-conjugate Normal Normal x InverseGamma hierarchy for univariate data.
 *
 * This class represents a hierarchical model where data are distributed
 * according to a Normal likelihood (see the `UniNormLikelihood` class for
 * details). The likelihood parameters have a Normal x InverseGamma centering
 * distribution (see the `NxIGPriorModel` class for details). That is:
 *
 * \f[
 *    f(x_i \mid \mu,\sigma^2) &= N(\mu,\sigma^2) \\
 *    \mu &\sim N(\mu_0, \eta^2) \\
 *    \sigma^2 &\sim InvGamma(a, b)
 * \f]
 *
 * The state is composed of mean and variance. The state hyperparameters are
 * \f$ (\mu_0, \eta^2, a, b) \f$, all scalar values. Note that this hierarchy
 * is NOT conjugate, meaning that the marginal distribution is not available
 * in closed form
 */

class NNxIGHierarchy
    : public BaseHierarchy<NNxIGHierarchy, UniNormLikelihood, NxIGPriorModel> {
 public:
  NNxIGHierarchy() = default;
  ~NNxIGHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNxIG;
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() { updater = std::make_shared<NNxIGUpdater>(); }

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
};

#endif  // BAYESMIX_HIERARCHIES_NNXIG_HIERARCHY_H_
