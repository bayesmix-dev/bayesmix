#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_

// #include <google/protobuf/stubs/casts.h>

// #include <Eigen/Dense>
// #include <memory>
// #include <vector>

// #include "algorithm_state.pb.h"
// #include "conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
// #include "hierarchy_prior.pb.h"

#include "base_hierarchy.h"
#include "likelihoods/uni_norm_likelihood.h"
#include "priors/nig_prior_model.h"
#include "updaters/nnig_updater.h"
#include "updaters/random_walk_updater.h"

class NNIGHierarchy : public BaseHierarchy<NNIGHierarchy, UniNormLikelihood,
                                           NIGPriorModel, RandomWalkUpdater> {
 public:
  NNIGHierarchy() = default;
  ~NNIGHierarchy() = default;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNIG;
  }

  double marg_lpdf(const HyperParams &params,
                   const Eigen::RowVectorXd &datum) const override {
    double sig_n = sqrt(params.scale * (params.var_scaling + 1) /
                        (params.shape * params.var_scaling));
    return stan::math::student_t_lpdf(datum(0), 2 * params.shape, params.mean,
                                      sig_n);
  }
};

#endif
