#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/uni_norm_likelihood.h"
#include "priors/nig_prior_model.h"
#include "updaters/nnig_updater.h"

class NNIGHierarchy
    : public BaseHierarchy<NNIGHierarchy, UniNormLikelihood, NIGPriorModel> {
 public:
  ~NNIGHierarchy() = default;

  using BaseHierarchy<NNIGHierarchy, UniNormLikelihood,
                      NIGPriorModel>::BaseHierarchy;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNIG;
  }

  void set_default_updater() { updater = std::make_shared<NNIGUpdater>(); }

  void initialize_state() override {
    // Get hypers
    auto hypers = prior->get_hypers();
    // Initialize likelihood state
    State::UniLS state;
    state.mean = hypers.mean;
    state.var = hypers.scale / (hypers.shape + 1);
    like->set_state(state);
  };

  double marg_lpdf(ProtoHypersPtr hier_params,
                   const Eigen::RowVectorXd &datum) const override {
    auto params = hier_params->nnig_state();
    double sig_n = sqrt(params.scale() * (params.var_scaling() + 1) /
                        (params.shape() * params.var_scaling()));
    return stan::math::student_t_lpdf(datum(0), 2 * params.shape(),
                                      params.mean(), sig_n);
  }
};

#endif
