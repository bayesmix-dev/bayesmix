#ifndef BAYESMIX_HIERARCHIES_BETA_GG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BETA_GG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/beta_likelihood.h"
#include "priors/gamma_gamma_prior.h"
#include "updaters/random_walk_updater.h"

/**
 *
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
