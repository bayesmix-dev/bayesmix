#ifndef BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/uni_lin_reg_likelihood.h"
#include "priors/mnig_prior_model.h"
#include "updaters/mnig_updater.h"

class LinRegUniHierarchy
    : public BaseHierarchy<LinRegUniHierarchy, UniLinRegLikelihood,
                           MNIGPriorModel> {
 public:
  ~LinRegUniHierarchy() = default;

  using BaseHierarchy<LinRegUniHierarchy, UniLinRegLikelihood,
                      MNIGPriorModel>::BaseHierarchy;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LinRegUni;
  }

  void set_default_updater() { updater = std::make_shared<MNIGUpdater>(); }

  void initialize_state() override {
    // Initialize likelihood dimension to prior one
    like->set_dim(prior->get_dim());
    // Get hypers
    auto hypers = prior->get_hypers();
    // Initialize likelihood state
    State::UniLinRegLS state;
    state.regression_coeffs = hypers.mean;
    state.var = hypers.scale / (hypers.shape + 1);
    like->set_state(state);
  };

  double marg_lpdf(ProtoHypersPtr hier_params, const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const override {
    auto params = hier_params->lin_reg_uni_state();
    Eigen::VectorXd mean = bayesmix::to_eigen(params.mean());
    Eigen::MatrixXd var_scaling = bayesmix::to_eigen(params.var_scaling());

    auto I = Eigen::MatrixXd::Identity(prior->get_dim(), prior->get_dim());
    Eigen::MatrixXd var_scaling_inv = var_scaling.llt().solve(I);

    double sig_n =
        sqrt((1 + (covariate * var_scaling_inv * covariate.transpose())(0)) *
             params.scale() / params.shape());
    return stan::math::student_t_lpdf(datum(0), 2 * params.shape(),
                                      covariate.dot(mean), sig_n);
  };
};

#endif  // BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
