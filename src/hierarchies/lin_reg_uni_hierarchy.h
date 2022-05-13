#ifndef BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/uni_lin_reg_likelihood.h"
#include "priors/mnig_prior_model.h"
#include "updaters/mnig_updater.h"

//! Linear regression hierarchy for univariate data.
//!
//! This class implements a dependent hierarchy which represents the classical
//! univariate Bayesian linear regression model, i.e.:
//!    y_i | \beta, x_i, \sigma^2 \sim N(\beta^T x_i, sigma^2)
//!              \beta | \sigma^2 \sim N(\mu, sigma^2 Lambda^{-1})
//!                      \sigma^2 \sim InvGamma(a, b)
//!
//! The state consists of the `regression_coeffs` \beta, and the `var` sigma^2.
//! Lambda is called the variance-scaling factor. Note that this hierarchy is
//! conjugate, thus the marginal distribution is available in closed form. For
//! more information, please refer to parent classes: `BaseHierarchy`,
//! `UniLinRegLikelihood` for deatails on the likelihood model, and
//! `MNIGPriorModel` for details on the prior model.

class LinRegUniHierarchy
    : public BaseHierarchy<LinRegUniHierarchy, UniLinRegLikelihood,
                           MNIGPriorModel> {
 public:
  LinRegUniHierarchy() = default;
  ~LinRegUniHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LinRegUni;
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() { updater = std::make_shared<MNIGUpdater>(); }

  //! Initializes state parameters to appropriate values
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

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
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
