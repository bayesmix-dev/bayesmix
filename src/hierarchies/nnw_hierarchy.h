#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_state.pb.h"
#include "conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

//! Normal Normal-Wishart hierarchy for multivariate data.

//! This class represents a hierarchy, i.e. a cluster, whose multivariate data
//! are distributed according to a multinomial normal likelihood, the
//! parameters of which have a Normal-Wishart centering distribution. That is:
//!           phi = (mu,tau)   (state);
//! f(x_i|mu,tau) = N(mu,tau)  (data likelihood);
//!      (mu,tau) ~ G          (unique values distribution);
//!             G ~ MM         (mixture model);
//!            G0 = N-W        (centering distribution).
//! The state is composed of mean and precision matrix. The Cholesky factor and
//! log-determinant of the latter are also included in the container for
//! efficiency reasons. The state's hyperparameters, contained in the Hypers
//! object, are (mu0, lambda0, tau0, nu0), which are respectively vector,
//! scalar, matrix, and scalar. Note that this hierarchy is conjugate, thus the
//! marginal distribution is available in closed form.  For more information,
//! please refer to parent classes: `AbstractHierarchy`, `BaseHierarchy`, and
//! `ConjugateHierarchy`.

namespace NNW {
//! Custom container for State values
struct State {
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec;
  Eigen::MatrixXd prec_chol;
  double prec_logdet;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  Eigen::VectorXd mean;
  double var_scaling;
  double deg_free;
  Eigen::MatrixXd scale;
  Eigen::MatrixXd scale_inv;
  Eigen::MatrixXd scale_chol;
};
}  // namespace NNW

class NNWHierarchy
    : public ConjugateHierarchy<NNWHierarchy, NNW::State, NNW::Hyperparams,
                                bayesmix::NNWPrior> {
 public:
  NNWHierarchy() = default;
  ~NNWHierarchy() = default;

  double like_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate =
                       Eigen::RowVectorXd(0)) const override;

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double marg_lpdf(
      const NNW::Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const;

  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data,
                                 const Eigen::MatrixXd &covariates =
                                     Eigen::MatrixXd(0, 0)) const override;

  Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  void initialize_state();

  void initialize_hypers();

  void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  NNW::State draw(const NNW::Hyperparams &params);

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param covariate  Covariate vector associated to datum
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate,
                                 bool add);

  //! Removes every data point from this cluster
  void clear_data();

  bool is_multivariate() const override { return true; }

  //! Computes and return posterior hypers given data currently in this cluster
  NNW::Hyperparams get_posterior_parameters() const;

  void set_state_from_proto(const google::protobuf::Message &state_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNW;
  }

 protected:
  //! Writes prec and its utilities to the given state object by pointer
  void write_prec_to_state(const Eigen::MatrixXd &prec_, NNW::State *out);
  //! Returns parameters for the predictive Student's t distribution
  NNW::Hyperparams get_predictive_t_parameters(
      const NNW::Hyperparams &params) const;

  //! Dimension of data space
  unsigned int dim;
  //! Sum of data points currently belonging to the cluster
  Eigen::VectorXd data_sum;
  //! Sum of squared data points currently belonging to the cluster
  Eigen::MatrixXd data_sum_squares;
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
