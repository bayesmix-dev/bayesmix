#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_BASE_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_BASE_LIKELIHOOD_H_

#include <google/protobuf/message.h>

#include <memory>
#include <set>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "abstract_likelihood.h"
#include "algorithm_state.pb.h"
#include "likelihood_internal.h"
#include "src/utils/covariates_getter.h"
#include "src/utils/rng.h"

//! Base template class of a `Likelihood` object
//!
//! This class derives from `AbstractLikelihood` and is templated over
//! `Derived` (needed for the curiously recurring template pattern) and
//! `State`: an instance of `BaseState`
//!
//! @tparam Derived  Name of the implemented derived class
//! @tparam State    Class name of the container for state values

template <class Derived, typename State>
class BaseLikelihood : public AbstractLikelihood {
 public:
  //! Default constructor
  BaseLikelihood() = default;

  //! Default destructor
  ~BaseLikelihood() = default;

  //! Returns an independent, data-less copy of this object
  std::shared_ptr<AbstractLikelihood> clone() const override {
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    out->clear_data();
    out->clear_summary_statistics();
    return out;
  }

  virtual Eigen::VectorXd sample() const override {
    throw std::runtime_error("sample() not yet implemented");
  }

  //! Evaluates the log likelihood over all the data in the cluster
  //! given unconstrained parameter values.
  //! By unconstrained parameters we mean that each entry of
  //! the parameter vector can range over (-inf, inf).
  //! Usually, some kind of transformation is required from the unconstrained
  //! parametrization to the actual one.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood over all data in the cluster
  double cluster_lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) const override {
    return internal::cluster_lpdf_from_unconstrained(
        static_cast<const Derived &>(*this), unconstrained_params, 0);
  }

  //! This version using `stan::math::var` type is required for Stan automatic
  //! differentiation. Evaluates the log likelihood over all the data in the
  //! cluster given unconstrained parameter values. By unconstrained parameters
  //! we mean that each entry of the parameter vector can range over (-inf,
  //! inf). Usually, some kind of transformation is required from the
  //! unconstrained parametrization to the actual one.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood over all data in the cluster
  stan::math::var cluster_lpdf_from_unconstrained(
      Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> unconstrained_params)
      const override {
    return internal::cluster_lpdf_from_unconstrained(
        static_cast<const Derived &>(*this), unconstrained_params, 0);
  }

  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  Eigen::VectorXd lpdf_grid(const Eigen::MatrixXd &data,
                            const Eigen::MatrixXd &covariates =
                                Eigen::MatrixXd(0, 0)) const override;

  //! Returns the current cardinality of the cluster
  int get_card() const { return card; }

  //! Returns the logarithm of the current cardinality of the cluster
  double get_log_card() const { return log_card; }

  //! Returns the indexes of data points belonging to this cluster
  std::set<int> get_data_idx() const { return cluster_data_idx; }

  //! Writes current state to a Protobuf message by pointer
  void write_state_to_proto(google::protobuf::Message *out) const override;

  //! Returns the class of the current state for the likelihood
  State get_state() const { return state; }

  State *mutable_state() { return &state; }

  //! Returns a vector storing the state in its unconstrained form
  Eigen::VectorXd get_unconstrained_state() override {
    return internal::get_unconstrained_state(state, 0);
  }

  //! Updates the state of the likelihood with the object given as input
  void set_state(const State &state_, bool update_card = true) {
    state = state_;
    if (update_card) {
      set_card(state.card);
    }
  };

  void set_state_from_proto(const google::protobuf::Message &state_,
                            bool update_card = true) override {
    State new_state;
    new_state.set_from_proto(downcast_state(state_), update_card);
    set_state(new_state, update_card);
  }

  //! Updates the state of the likelihood starting from its unconstrained form
  void set_state_from_unconstrained(
      const Eigen::VectorXd &unconstrained_state) override {
    internal::set_state_from_unconstrained(state, unconstrained_state, 0);
  }

  //! Sets the (pointer to) the dataset in the cluster
  void set_dataset(const Eigen::MatrixXd *const dataset) override {
    dataset_ptr = dataset;
  }

  //! Returns the (pointer to) the dataset in the cluster
  const Eigen::MatrixXd *get_dataset() const { return dataset_ptr; }

  //! Adds a datum and its index to the likelihood
  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  //! Removes a datum and its index from the likelihood
  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  //! Resets cardinality and indexes of data in this cluster
  void clear_data() override {
    set_card(0);
    cluster_data_idx = std::set<int>();
  }

 protected:
  //! Sets the cardinality of the cluster
  void set_card(const int card_) {
    card = card_;
    log_card = (card_ == 0) ? stan::math::NEGATIVE_INFTY : std::log(card_);
  }

  //! Down-casts the given generic proto message to a ClusterState proto
  bayesmix::AlgorithmState::ClusterState *downcast_state(
      google::protobuf::Message *state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::ClusterState *>(state_);
  }

  //! Down-casts the given generic proto message to a ClusterState proto
  const bayesmix::AlgorithmState::ClusterState &downcast_state(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::ClusterState &>(state_);
  }

  //! Current state of this cluster
  State state;

  //! Current cardinality of this cluster
  int card = 0;

  //! Logarithm of current cardinality of this cluster
  double log_card = stan::math::NEGATIVE_INFTY;

  //! Set of indexes of data points belonging to this cluster
  std::set<int> cluster_data_idx;

  //! Pointer to the cluster dataset
  const Eigen::MatrixXd *dataset_ptr;
};

template <class Derived, typename State>
void BaseLikelihood<Derived, State>::add_datum(
    const int id, const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) {
  assert(cluster_data_idx.find(id) == cluster_data_idx.end());
  set_card(++card);
  static_cast<Derived *>(this)->update_summary_statistics(datum, covariate,
                                                          true);
  cluster_data_idx.insert(id);
}

template <class Derived, typename State>
void BaseLikelihood<Derived, State>::remove_datum(
    const int id, const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) {
  static_cast<Derived *>(this)->update_summary_statistics(datum, covariate,
                                                          false);
  set_card(--card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
}

template <class Derived, typename State>
void BaseLikelihood<Derived, State>::write_state_to_proto(
    google::protobuf::Message *out) const {
  auto *out_cast = downcast_state(out);
  out_cast->CopyFrom(state.get_as_proto());
  out_cast->set_cardinality(card);
}

template <class Derived, typename State>
Eigen::VectorXd BaseLikelihood<Derived, State>::lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd lpdf(data.rows());
  covariates_getter cov_getter(covariates);
  for (int i = 0; i < data.rows(); i++)
    lpdf(i) =
        static_cast<Derived const *>(this)->lpdf(data.row(i), cov_getter(i));
  return lpdf;
}

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_BASE_LIKELIHOOD_H_
