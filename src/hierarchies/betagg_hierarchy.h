#ifndef BAYESMIX_HIERARCHIES_BETA_GG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BETA_GG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/beta_likelihood.h"
#include "priors/gamma_gamma_prior.h"
#include "updaters/random_walk_updater.h"

/**
 * Beta Gamma-Gamma hierarchy for univaraite data in [0, 1]
 *
 * This class represents a hierarchical model where data are distributed
 * according to a Beta likelihood (see the `BetaLikelihood` class for
 * details). The shape and rate parameters of the likelihood have
 * independent gamma priors. That is
 *
 * \f[
 *    f(x_i \mid \alpha, \beta) &= Beta(\alpha, \beta) \\
 *    \alpha &\sim Gamma(\alpha_a, \alpha_b) \\
 *    \beta &\sim Gamma(\beta_a, \beta_b)
 * \f]
 *
 * The state is composed of shape and rate. Note that this hierarchy
 * is NOT conjugate, meaning that the marginal distribution is not available
 * in closed form.
 */

class BetaGGHierarchy
    : public BaseHierarchy<BetaGGHierarchy, BetaLikelihood, GGPriorModel> {
 public:
  BetaGGHierarchy() = default;
  ~BetaGGHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::BetaGG;
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() {
    updater = std::make_shared<RandomWalkUpdater>(0.1);
  }

  //! Initializes state parameters to appropriate values
  void initialize_state() override {
    // Get hypers
    auto hypers = prior->get_hypers();
    // Initialize likelihood state
    State::ShapeRate state;
    state.shape = hypers.a_shape / hypers.a_rate;
    state.rate = hypers.b_shape / hypers.b_rate;
    like->set_state(state);
  };
};

#endif  // BAYESMIX_HIERARCHIES_BETA_GG_HIERARCHY_H_
