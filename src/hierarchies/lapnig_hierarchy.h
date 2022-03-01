#ifndef BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_

// #include <google/protobuf/stubs/casts.h>

// #include <Eigen/Dense>
// #include <memory>
// #include <vector>

// #include "algorithm_state.pb.h"
// #include "conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
// #include "hierarchy_prior.pb.h"

#include "base_hierarchy.h"
#include "likelihoods/laplace_likelihood.h"
#include "priors/nxig_prior_model.h"
#include "updaters/mala_updater.h"

class LapNIGHierarchy
    : public BaseHierarchy<LapNIGHierarchy, LaplaceLikelihood,
                           NxIGPriorModel> {
 public:
  LapNIGHierarchy() = default;
  ~LapNIGHierarchy() = default;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LapNIG;
  }

  void set_default_updater() { updater = std::make_shared<MalaUpdater>(); }

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
