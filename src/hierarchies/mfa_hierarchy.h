#ifndef BAYESMIX_HIERARCHIES_MFA_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_MFA_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

//! Mixture of Factor Analysers hierarchy for multivariate data.

//! This class represents a hierarchical model where data are distributed

namespace MFA {
//! Custom container for State values
struct State {
  Eigen::VectorXd mu, psi;
  Eigen::MatrixXd Eta, Lambda;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  Eigen::VectorXd mutilde, beta;
  double phi, alpha0;
  unsigned int q;
};

};  // namespace MFA

class MFAHierarchy
    : public BaseHierarchy<MFAHierarchy, MFA::State, MFA::Hyperparams,
                           bayesmix::MFAPrior> {
 public:
  MFAHierarchy() = default;
  ~MFAHierarchy() = default;

  //! Updates hyperparameter values given a vector of cluster states
  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>&
                         states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  MFA::State draw(const MFA::Hyperparams& params);

  //! Resets summary statistics for this cluster
  void clear_summary_statistics() override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::MFA;
  }

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message& state_) override;

  //! Read and set hyperparameter values from a given Protobuf message
  void set_hypers_from_proto(
      const google::protobuf::Message& hypers_) override;

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

  //! Returns whether the hierarchy models multivariate data or not
  bool is_multivariate() const override { return true; }

  //! Saves posterior hyperparameters to the corresponding class member
  void save_posterior_hypers() {
    // TODO
    /* posterior_hypers =
        static_cast<Derived *>(this)->compute_posterior_hypers();*/
  }

  //! Generates new state values from the centering posterior distribution
  //! @param update_params  Save posterior hypers after the computation?
  void sample_full_cond(bool update_params = false) override;

 protected:
  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double like_lpdf(const Eigen::RowVectorXd& datum) const override;

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd& datum,
                                 bool add) override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;

  //! Initializes hierarchy hyperparameters to appropriate values
  void initialize_hypers() override;

  void sample_Eta();

  void sample_mu();

  void sample_psi();

  void sample_Lambda();

  //! Sum of data points currently belonging to the cluster
  Eigen::VectorXd data_sum;

  //! Number of initial variables
  size_t p;

  //! Vector of data points currently belonging to the cluster
  std::vector<Eigen::VectorXd> data;
};

#endif  // BAYESMIX_HIERARCHIES_MFA_HIERARCHY_H_
