#ifndef BAYESMIX_HIERARCHIES_GAMMA_GAMMA_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_GAMMA_GAMMA_HIERARCHY_H_

#include "gamma_likelihood.h"
#include "gamma_prior_model.h"
#include "gammagamma_updater.h"
#include "hierarchy_id.pb.h"
#include "src/hierarchies/base_hierarchy.h"

class GammaGammaHierarchy
    : public BaseHierarchy<GammaGammaHierarchy, GammaLikelihood,
                           GammaPriorModel> {
 public:
  GammaGammaHierarchy(double shape_, double rate_alpha_, double rate_beta_) {
    auto prior =
        std::make_shared<GammaPriorModel>(shape_, rate_alpha_, rate_beta_);
    set_prior(prior);
  };
  ~GammaGammaHierarchy() = default;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::UNKNOWN_HIERARCHY;
  }

  void set_default_updater() {
    updater = std::make_shared<GammaGammaUpdater>();
  }

  void initialize_state() override {
    // Get hypers
    auto hypers = prior->get_hypers();
    // Initialize likelihood state
    State::Gamma state;
    state.shape = prior->get_shape();
    state.rate = hypers.rate_alpha / hypers.rate_beta;
    like->set_state(state);
  };

  double marg_lpdf(ProtoHypersPtr hier_params,
                   const Eigen::RowVectorXd &datum) const override {
    throw(
        std::runtime_error("marg_lpdf() not implemented for this hierarchy"));
    return 0;
  }
};

#endif  // BAYESMIX_HIERARCHIES_GAMMA_GAMMA_HIERARCHY_H_
