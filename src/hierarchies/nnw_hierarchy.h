#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/multi_norm_likelihood.h"
#include "priors/nw_prior_model.h"
#include "src/utils/distributions.h"
#include "updaters/nnw_updater.h"

/**
 * Normal Normal-Wishart hierarchy for multivariate data.
 *
 * This class represents a hierarchy whose multivariate data
 * are distributed according to a multivariate normal likelihood (see the
 * `MultiNormLikelihood` for details). The likelihood parameters have a
 * Normal-Wishart centering distribution (see the `NWPriorModel` class for
 * details). That is:
 *
 * \f[
 *    f(\bm{x}_i \mid \bm{\mu},\Sigma) &= N_d(\bm{\mu},\Sigma^{-1}) \\
 *    (\bm{\mu},\Sigma) &\sim NW(\mu_0, \lambda, \Psi_0, \nu_0)
 * \f]
 * The state is composed of mean and precision matrix. The Cholesky factor and
 * log-determinant of the latter are also included in the container for
 * efficiency reasons. The state's hyperparameters are \f$(\mu_0, \lambda,
 * \Psi_0, \nu_0)\f$, which are respectively vector, scalar, matrix, and
 * scalar. Note that this hierarchy is conjugate, thus the marginal
 * distribution is available in closed form
 */

class NNWHierarchy
    : public BaseHierarchy<NNWHierarchy, MultiNormLikelihood, NWPriorModel> {
 public:
  NNWHierarchy() = default;
  ~NNWHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNW;
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() { updater = std::make_shared<NNWUpdater>(); }

  //! Initializes state parameters to appropriate values
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

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param hier_params  Pointer to the container of (prior or posterior)
  //! hyperparameter values
  //! @param datum        Point which is to be evaluated
  //! @return             The evaluation of the lpdf
  double marg_lpdf(ProtoHypersPtr hier_params,
                   const Eigen::RowVectorXd &datum) const override {
    HyperParams pred_params = get_predictive_t_parameters(hier_params);
    Eigen::VectorXd diag = pred_params.scale_chol.diagonal();
    double logdet = 2 * log(diag.array()).sum();
    return bayesmix::multi_student_t_invscale_lpdf(
        datum, pred_params.deg_free, pred_params.mean, pred_params.scale_chol,
        logdet);
  }

  //! Helper function that computes the predictive parameters for the
  //! multivariate t distribution from the current hyperparameter values. It is
  //! used to efficiently compute the log-marginal distribution of data.
  //! @param hier_params  Pointer to the container of (prior or posterior)
  //! hyperparameter values
  //! @return             A `HyperParam` object with the predictive parameters
  HyperParams get_predictive_t_parameters(ProtoHypersPtr hier_params) const {
    auto params = hier_params->nnw_state();
    // Compute dof and scale of marginal distribution
    unsigned int dim = like->get_dim();
    double nu_n = params.deg_free() - dim + 1;
    double coeff = (params.var_scaling() + 1) / (params.var_scaling() * nu_n);
    Eigen::MatrixXd scale_chol =
        Eigen::LLT<Eigen::MatrixXd>(bayesmix::to_eigen(params.scale()))
            .matrixU();
    Eigen::MatrixXd scale_chol_n = scale_chol / std::sqrt(coeff);
    // Return predictive t parameters
    HyperParams out;
    out.mean = bayesmix::to_eigen(params.mean());
    out.deg_free = nu_n;
    out.scale_chol = scale_chol_n;
    return out;
  }
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
