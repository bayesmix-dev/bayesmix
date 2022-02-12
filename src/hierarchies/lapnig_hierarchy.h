#ifndef BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <set>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

//! Laplace Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchical model where data are distributed
//! according to a laplace likelihood, the parameters of which have a
//! Normal-InverseGamma centering distribution. That is:
//! f(x_i|mu,lambda) = Laplace(mu,lambda)
//!    (mu,lambda) ~ N-IG(mu0, lambda0, alpha0, beta0)
//! The state is composed of mean and scale. The state hyperparameters,
//! contained in the Hypers object, are (mu_0, lambda0, alpha0, beta0,
//! scale_var, mean_var), all scalar values. Note that this hierarchy is NOT
//! conjugate, thus the marginal distribution is not available in closed form.
//! The hyperprameters scale_var and mean_var are used to perform a step of
//! Random Walk Metropolis Hastings to sample from the full conditionals. For
//! more information, please refer to parent classes: `AbstractHierarchy` and
//! `BaseHierarchy`.

namespace LapNIG {
//! Custom container for State values
struct State {
  double mean, scale;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  double mean, var, shape, scale, mh_log_scale_var, mh_mean_var;
};
}  // namespace LapNIG

class LapNIGHierarchy
    : public BaseHierarchy<LapNIGHierarchy, LapNIG::State, LapNIG::Hyperparams,
                           bayesmix::LapNIGPrior> {
 public:
  LapNIGHierarchy() = default;
  ~LapNIGHierarchy() = default;

  //! Counters for tracking acceptance rate in MH step
  static unsigned int accepted_;
  static unsigned int iter_;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LapNIG;
  }

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Read and set hyperparameter values from a given Protobuf message
  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  void save_posterior_hypers() { throw std::runtime_error("Not implemented"); }

  //! Returns whether the hierarchy models multivariate data or not
  bool is_multivariate() const override { return false; }

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::ClusterState message by adding the appropriate type
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;

  //! Writes current value of hyperparameters to a Protobuf message and
  //! return a shared_ptr.
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::HierarchyHypers message by adding the appropriate type
  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override;

  //! Resets summary statistics for this cluster
  void clear_summary_statistics() override;

  //! Updates hyperparameter values given a vector of cluster states
  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  LapNIG::State draw(const LapNIG::Hyperparams &params);

  //! Generates new state values from the centering posterior distribution
  //! @param update_params  Save posterior hypers after the computation?
  void sample_full_cond(bool update_params = false) override;

 protected:
  //! Set of values of data points belonging to this cluster
  std::list<Eigen::RowVectorXd> cluster_data_values;

  //! Sum of absolute differences for current params
  double sum_abs_diff_curr = 0;

  //! Sum of absolute differences for proposal params
  double sum_abs_diff_prop = 0;

  //! Samples from the proposal distribution using Random Walk
  //! Metropolis-Hastings
  Eigen::VectorXd propose_rwmh(const Eigen::VectorXd &curr_vals);

  //! Evaluates the prior given the mean (unconstrained_parameters(0))
  //! and log of the scale (unconstrained_parameters(1))
  double eval_prior_lpdf_unconstrained(
      Eigen::VectorXd unconstrained_parameters);

  //! Evaluates the (sum of the) log likelihood for all the observations in the
  //! cluster given the mean (unconstrained_parameters(0))
  //! and log of the scale (unconstrained_parameters(1)).
  //! The parameter "is_current" is used to identify if the evaluation of the
  //! likelihood is on the current or on the proposed parameters, in order to
  //! avoid repeating calculations of the sum of the absolute differences
  double eval_like_lpdf_unconstrained(Eigen::VectorXd unconstrained_parameters,
                                      bool is_current);

  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double like_lpdf(const Eigen::RowVectorXd &datum) const override;

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 bool add) override;

  //! Initializes hierarchy hyperparameters to appropriate values
  void initialize_hypers() override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;
};
#endif  // BAYESMIX_HIERARCHIES_LAPNIG_HIERARCHY_H_
