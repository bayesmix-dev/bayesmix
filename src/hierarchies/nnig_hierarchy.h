#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_state.pb.h"
#include "conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

//! Conjugate Normal Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchical model where data are distributed
//! according to a normal likelihood, the parameters of which have a
//! Normal-InverseGamma centering distribution. That is:
//! f(x_i|mu,sig) = N(mu,sig^2)
//!    (mu,sig^2) ~ N-IG(mu0, lambda0, alpha0, beta0)
//! The state is composed of mean and variance. The state hyperparameters,
//! contained in the Hypers object, are (mu_0, lambda0, alpha0, beta0), all
//! scalar values. Note that this hierarchy is conjugate, thus the marginal
//! distribution is available in closed form.  For more information, please
//! refer to parent classes: `AbstractHierarchy`, `BaseHierarchy`, and
//! `ConjugateHierarchy`.

namespace NNIG {
//! Custom container for State values
struct State {
  double mean, var;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  double mean, var_scaling, shape, scale;
};

};  // namespace NNIG

class NNIGHierarchy
    : public ConjugateHierarchy<NNIGHierarchy, NNIG::State, NNIG::Hyperparams,
                                bayesmix::NNIGPrior> {
 public:
  NNIGHierarchy() = default;
  ~NNIGHierarchy() = default;

  double like_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate =
                       Eigen::RowVectorXd(0)) const override;

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double marg_lpdf(
      const NNIG::Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const;

  void initialize_state() override;

  void initialize_hypers();

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  NNIG::State draw(const NNIG::Hyperparams &params);

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param covariate  Covariate vector associated to datum
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate,
                                 bool add);

  //! Removes every data point from this cluster
  void clear_summary_statistics();

  bool is_multivariate() const override { return false; }

  //! Computes and return posterior hypers given data currently in this cluster
  NNIG::Hyperparams get_posterior_parameters();

  void set_state_from_proto(const google::protobuf::Message &state_) override;

  std::shared_ptr<google::protobuf::Message> get_state_proto() const override;

  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNIG;
  }

 protected:
  //! Sum of data points currently belonging to the cluster
  double data_sum = 0;

  //! Sum of squared data points currently belonging to the cluster
  double data_sum_squares = 0;
};

#endif  // BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_H_
