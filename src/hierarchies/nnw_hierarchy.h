#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_

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
#include "likelihoods/multi_norm_likelihood.h"
#include "priors/nw_prior_model.h"
#include "updaters/nnw_updater.h"

class NNWHierarchy
    : public BaseHierarchy<NNWHierarchy, MultiNormLikelihood, NWPriorModel> {
 public:
  NNWHierarchy() = default;
  ~NNWHierarchy() = default;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNW;
  }

  void set_default_updater() { updater = std::make_shared<NNWUpdater>(); }

  void initialize_state() override {
    // Initialize likelihood dimension to prior one
    like->set_dim(prior->get_dim());
    // Get hypers and data dimension
    auto hypers = prior->get_hypers();
    unsigned int dim = like->get_dim();
    // Initialize likelihood state
    State::MultiLS state;
    state.mean = hypers.mean;
    prior->write_prec_to_state(
        hypers.var_scaling * Eigen::MatrixXd::Identity(dim, dim), &state);
    like->set_state(state);
  };

  double marg_lpdf(const HyperParams &params,
                   const Eigen::RowVectorXd &datum) const override {
    HyperParams pred_params = get_predictive_t_parameters(params);
    Eigen::VectorXd diag = pred_params.scale_chol.diagonal();
    double logdet = 2 * log(diag.array()).sum();
    return bayesmix::multi_student_t_invscale_lpdf(
        datum, pred_params.deg_free, pred_params.mean, pred_params.scale_chol,
        logdet);
  }

  HyperParams get_predictive_t_parameters(const HyperParams &params) const {
    // Compute dof and scale of marginal distribution
    unsigned int dim = like->get_dim();
    double nu_n = params.deg_free - dim + 1;
    double coeff = (params.var_scaling + 1) / (params.var_scaling * nu_n);
    Eigen::MatrixXd scale_chol_n = params.scale_chol / std::sqrt(coeff);
    // Return predictive t parameters
    HyperParams out;
    out.mean = params.mean;
    out.deg_free = nu_n;
    out.scale_chol = scale_chol_n;
    return out;
  }
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
