#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "abstract_hierarchy.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "src/utils/rng.h"

//! Base template class for a hierarchy object.

//! This class is a templatized version of, and derived from, the
//! `AbstractHierarchy` class, and the second stage of the curiously recurring
//! template pattern for `Hierarchy` objects (please see the docs of the parent
//! class for further information). It includes class members and some more
//! functions which could not be implemented in the non-templatized abstract
//! class.
//! See, for instance, `ConjugateHierarchy` and `NNIGHierarchy` to better
//! understand the CRTP patterns.

//! @tparam Derived      Name of the implemented derived class
//! @tparam State        Class name of the container for state values
//! @tparam Hyperparams  Class name of the container for hyperprior parameters
//! @tparam Prior        Class name of the container for prior parameters

template <class Derived, typename State, typename Hyperparams, typename Prior>
class BaseHierarchy : public AbstractHierarchy {
 public:
  BaseHierarchy() = default;
  ~BaseHierarchy() = default;

  //! Returns an independent, data-less copy of this object
  virtual std::shared_ptr<AbstractHierarchy> clone() const override {
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    out->clear_data();
    out->clear_summary_statistics();
    return out;
  }

  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  //! Generates new state values from the centering prior distribution
  void sample_prior() override {
    state = static_cast<Derived *>(this)->draw(*hypers);
  }

  //! Overloaded version of sample_full_cond(bool), mainly used for debugging
  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

  //! Returns the current cardinality of the cluster
  int get_card() const override { return card; }

  //! Returns the logarithm of the current cardinality of the cluster
  double get_log_card() const override { return log_card; }

  //! Returns the indexes of data points belonging to this cluster
  std::set<int> get_data_idx() const override { return cluster_data_idx; }

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  virtual google::protobuf::Message *get_mutable_prior() override {
    if (prior == nullptr) {
      create_empty_prior();
    }
    return prior.get();
  }

  //! Writes current state to a Protobuf message by pointer
  void write_state_to_proto(
      google::protobuf::Message *const out) const override;

  //! Writes current values of the hyperparameters to a Protobuf message by
  //! pointer
  void write_hypers_to_proto(
      google::protobuf::Message *const out) const override;

  //! Returns the struct of the current state
  State get_state() const { return state; }

  //! Returns the struct of the current prior hyperparameters
  Hyperparams get_hypers() const { return *hypers; }

  //! Returns the struct of the current posterior hyperparameters
  Hyperparams get_posterior_hypers() const { return posterior_hypers; }

  //! Adds a datum and its index to the hierarchy
  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  //! Removes a datum and its index from the hierarchy
  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  //! Main function that initializes members to appropriate values
  void initialize() override {
    hypers = std::make_shared<Hyperparams>();
    check_prior_is_set();
    initialize_hypers();
    initialize_state();
    posterior_hypers = *hypers;
    clear_data();
    clear_summary_statistics();
  }

  //! Sets the (pointer to the) dataset matrix
  void set_dataset(const Eigen::MatrixXd *const dataset) override {
    dataset_ptr = dataset;
  }

 protected:
  //! Raises an error if the prior pointer is not initialized
  void check_prior_is_set() const {
    if (prior == nullptr) {
      throw std::invalid_argument("Hierarchy prior was not provided");
    }
  }

  //! Re-initializes the prior of the hierarchy to a newly created object
  void create_empty_prior() { prior.reset(new Prior); }

  //! Sets the cardinality of the cluster
  void set_card(const int card_) {
    card = card_;
    log_card = (card_ == 0) ? stan::math::NEGATIVE_INFTY : std::log(card_);
  }

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::ClusterState message by adding the appropriate type
  virtual std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
  get_state_proto() const = 0;

  //! Initializes state parameters to appropriate values
  virtual void initialize_state() = 0;

  //! Writes current value of hyperparameters to a Protobuf message and
  //! return a shared_ptr.
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::HierarchyHypers message by adding the appropriate type
  virtual std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
  get_hypers_proto() const = 0;

  //! Initializes hierarchy hyperparameters to appropriate values
  virtual void initialize_hypers() = 0;

  //! Resets cardinality and indexes of data in this cluster
  void clear_data() {
    set_card(0);
    cluster_data_idx = std::set<int>();
  }

  virtual void clear_summary_statistics() = 0;

  //! Down-casts the given generic proto message to a ClusterState proto
  bayesmix::AlgorithmState::ClusterState *downcast_state(
      google::protobuf::Message *const state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::ClusterState *>(state_);
  }

  //! Down-casts the given generic proto message to a ClusterState proto
  const bayesmix::AlgorithmState::ClusterState &downcast_state(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::ClusterState &>(state_);
  }

  //! Down-casts the given generic proto message to a HierarchyHypers proto
  bayesmix::AlgorithmState::HierarchyHypers *downcast_hypers(
      google::protobuf::Message *const state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::HierarchyHypers *>(state_);
  }

  //! Down-casts the given generic proto message to a HierarchyHypers proto
  const bayesmix::AlgorithmState::HierarchyHypers &downcast_hypers(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::HierarchyHypers &>(state_);
  }

  //! Container for state values
  State state;

  //! Container for prior hyperparameters values
  std::shared_ptr<Hyperparams> hypers;

  //! Container for posterior hyperparameters values
  Hyperparams posterior_hypers;

  //! Pointer to a Protobuf prior object for this class
  std::shared_ptr<Prior> prior;

  //! Set of indexes of data points belonging to this cluster
  std::set<int> cluster_data_idx;

  //! Current cardinality of this cluster
  int card = 0;

  //! Logarithm of current cardinality of this cluster
  double log_card = stan::math::NEGATIVE_INFTY;

  //! Pointer to the dataset matrix for the mixture model
  const Eigen::MatrixXd *dataset_ptr = nullptr;
};

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::add_datum(
    const int id, const Eigen::RowVectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) {
  assert(cluster_data_idx.find(id) == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  static_cast<Derived *>(this)->update_ss(datum, covariate, true);
  cluster_data_idx.insert(id);
  if (update_params) {
    static_cast<Derived *>(this)->save_posterior_hypers();
  }
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::remove_datum(
    const int id, const Eigen::RowVectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::RowVectorXd &covariate /* = Eigen::RowVectorXd(0)*/) {
  static_cast<Derived *>(this)->update_ss(datum, covariate, false);
  set_card(card - 1);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
  if (update_params) {
    static_cast<Derived *>(this)->save_posterior_hypers();
  }
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::write_state_to_proto(
    google::protobuf::Message *const out) const {
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> state_ =
      get_state_proto();
  auto *out_cast = downcast_state(out);
  out_cast->CopyFrom(*state_.get());
  out_cast->set_cardinality(card);
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::write_hypers_to_proto(
    google::protobuf::Message *const out) const {
  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> hypers_ =
      get_hypers_proto();
  auto *out_cast = downcast_hypers(out);
  out_cast->CopyFrom(*hypers_.get());
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
BaseHierarchy<Derived, State, Hyperparams, Prior>::like_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    // Pass null value as covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->get_like_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else if (covariates.rows() == 1) {
    // Use unique covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->get_like_lpdf(
          data.row(i), covariates.row(0));
    }
  } else {
    // Use different covariates
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->get_like_lpdf(
          data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::sample_full_cond(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
  clear_data();
  clear_summary_statistics();
  if (covariates.cols() == 0) {
    // Pass null value as covariate
    for (int i = 0; i < data.rows(); i++) {
      static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                              Eigen::RowVectorXd(0));
    }
  } else if (covariates.rows() == 1) {
    // Use unique covariate
    for (int i = 0; i < data.rows(); i++) {
      static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                              covariates.row(0));
    }
  } else {
    // Use different covariates
    for (int i = 0; i < data.rows(); i++) {
      static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                              covariates.row(i));
    }
  }
  static_cast<Derived *>(this)->sample_full_cond(true);
}

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
