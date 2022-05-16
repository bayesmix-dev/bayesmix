#ifndef BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/laplace_likelihood.h"
#include "priors/nxig_prior_model.h"
#include "updaters/mala_updater.h"

//! Laplace Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchical model where data are distributed
//! according to a laplace likelihood (see the `LaplaceLikelihood` class for
//! deatils).The likelihood parameters have a Normal x InverseGamma centering
//! distribution (see the `NxIGPriorModel` class for details). That is:
//! \f[
//! f(x_i|\mu,\sigma^2) &= Laplace(\mu,\sqrt(\sigma^2/2))\\
//!               \mu &\sim N(\mu_0,\eta^2) \\
//!              \sigma^2 ~ IG(a, b)
//! \f]
//! The state is composed of mean and variance (thus the scale for the Laplace
//! distribution is \f$ \sqrt(\sigma^2 / 2)) \f$. The state hyperparameters are
//! \f$(mu_0, \sigma^2, a, b)\f$, all scalar values. Note that this hierarchy
//! is NOT conjugate, thus the marginal distribution is not available in closed
//! form.

class LapNIGHierarchy
    : public BaseHierarchy<LapNIGHierarchy, LaplaceLikelihood,
                           NxIGPriorModel> {
 public:
  LapNIGHierarchy() = default;
  ~LapNIGHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LapNIG;
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() { updater = std::make_shared<MalaUpdater>(); }

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

#endif  // BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_
