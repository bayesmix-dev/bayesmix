#ifndef BAYESMIX_HIERARCHIES_FA_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_FA_HIERARCHY_H_

// #include <google/protobuf/stubs/casts.h>

// #include <Eigen/Dense>
// #include <memory>
// #include <vector>

// #include "algorithm_state.pb.h"
// #include "conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "src/utils/distributions.h"
// #include "hierarchy_prior.pb.h"

#include "base_hierarchy.h"
#include "likelihoods/fa_likelihood.h"
#include "priors/fa_prior_model.h"
#include "updaters/fa_updater.h"

class FAHierarchy
    : public BaseHierarchy<FAHierarchy, FALikelihood, FAPriorModel> {
 public:
  FAHierarchy() = default;
  ~FAHierarchy() = default;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::FA;
  }

  void set_default_updater() { updater = std::make_shared<FAUpdater>(); }

  void initialize_state() override {
    // Initialize likelihood dimension to prior one
    like->set_dim(prior->get_dim());
    // Get hypers and data dimension
    auto hypers = prior->get_hypers();
    unsigned int dim = like->get_dim();
    // Initialize likelihood state
    State::FA state;
    state.mu = hypers.mutilde;
    state.psi = hypers.beta / (hypers.alpha0 + 1.);
    state.eta = Eigen::MatrixXd::Zero(hypers.card, hypers.q);
    state.lambda = Eigen::MatrixXd::Zero(dim, hypers.q);
    state.psi_inverse = state.psi.cwiseInverse().asDiagonal();
    like->set_state(state);
    like->compute_wood_factors(state.cov_wood, state.cov_logdet, state.lambda,
                               state.psi_inverse);
  }
};

#endif  // BAYESMIX_HIERARCHIES_FA_HIERARCHY_H_
