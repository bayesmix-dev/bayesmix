
#ifndef BAYESMIX_LAPNIG_HIERARCHY_H
#define BAYESMIX_LAPNIG_HIERARCHY_H

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <set>

#include "algorithm_state.pb.h"
#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"


namespace LapNIG {
//! Custom container for State values
struct State {
  double mean, scale;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  double mean, var, shape, scale;
};
}


class LapNIGHierarchy
    : public BaseHierarchy<LapNIGHierarchy, LapNIG::State, LapNIG::Hyperparams,
                           bayesmix::LapNIGPrior> { // Prior ? //
 public:
  //! Counters for tracking acceptance rate in MH step
  static unsigned int accepted_;
  static unsigned int iter_;

  LapNIGHierarchy() = default;
  ~LapNIGHierarchy() = default;
  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LapNIG;
  }

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Read and set hyperparameter values from a given Protobuf message
  void set_hypers_from_proto(
      const google::protobuf::Message &hypers_) override;

  void save_posterior_hypers() {
    throw std::runtime_error("Not implemented");
  }

  //! Returns whether the hierarchy models multivariate data or not
  bool is_multivariate() const override { return false; }

  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double like_lpdf(const Eigen::RowVectorXd &datum) const override;

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 bool add) override;

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

  //! Initializes hierarchy hyperparameters to appropriate values
  void initialize_hypers() override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;


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

};
#endif  // BAYESMIX_LAPNIG_HIERARCHY_H