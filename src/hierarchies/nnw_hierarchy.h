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
//! f(x_i|mu,tau) = N(mu,tau^{-1})
//!      (mu,tau) ~ NW(mu0, lambda0, tau0, nu0)
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

  // EVALUATION FUNCTIONS FOR GRIDS OF POINTS
  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data,
                                 const Eigen::MatrixXd &covariates =
                                     Eigen::MatrixXd(0, 0)) const override;

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  //! Updates hyperparameter values given a vector of cluster states
  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  NNW::State draw(const NNW::Hyperparams &params);

  //! Resets summary statistics for this cluster
  void clear_summary_statistics() override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNW;
  }

  //! Computes and return posterior hypers given data currently in this cluster
  NNW::Hyperparams get_posterior_parameters() const;

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::ClusterState message by adding the appropriate type
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;

  //! Writes current state to a Protobuf message by pointer
  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  //! Returns whether the hierarchy models multivariate data or not
  bool is_multivariate() const override { return true; }

 protected:
  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double like_lpdf(const Eigen::RowVectorXd &datum) const override;

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double marg_lpdf(const NNW::Hyperparams &params,
                   const Eigen::RowVectorXd &datum) const override;

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 bool add) override;

  //! Writes prec and its utilities to the given state object by pointer
  void write_prec_to_state(const Eigen::MatrixXd &prec_, NNW::State *out);

  //! Returns parameters for the predictive Student's t distribution
  NNW::Hyperparams get_predictive_t_parameters(
      const NNW::Hyperparams &params) const;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;

  //! Initializes hierarchy hyperparameters to appropriate values
  void initialize_hypers() override;

  //! Dimension of data space
  unsigned int dim;

  //! Sum of data points currently belonging to the cluster
  Eigen::VectorXd data_sum;

  //! Sum of squared data points currently belonging to the cluster
  Eigen::MatrixXd data_sum_squares;
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
