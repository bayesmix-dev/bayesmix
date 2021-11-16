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

  virtual std::shared_ptr<AbstractHierarchy> clone() const override {
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    out->clear_data();
    return out;
  }

  void sample_prior() override {
    state = static_cast<Derived *>(this)->draw(*hypers);
  };

  void initialize() override {
    hypers = std::make_shared<Hyperparams>();
    check_prior_is_set();
    static_cast<Derived *>(this)->initialize_hypers();
    static_cast<Derived *>(this)->initialize_state();
    posterior_hypers = *hypers;
    static_cast<Derived *>(this)->clear_data();
  }

  bool is_dependent() const override;

  void check_prior_is_set() const;

  virtual google::protobuf::Message *get_mutable_prior() override {
    if (prior == nullptr) create_empty_prior();

    return prior.get();
  }

  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  Hyperparams get_posterior_hypers() const { return posterior_hypers; }

  int get_card() const override { return card; }

  double get_log_card() const override { return log_card; }

  std::set<int> get_data_idx() const override { return cluster_data_idx; }

 protected:
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

  //! Initialize state parameters to appropriate values
  virtual void initialize_state() = 0;

  void create_empty_prior() { prior.reset(new Prior); }

  void set_card(const int card_) {
    card = card_;
    log_card = std::log(card_);
  }
};

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::add_datum(
    const int id, const Eigen::RowVectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) {
  assert(cluster_data_idx.find(id) == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  static_cast<Derived *>(this)->update_summary_statistics(datum, covariate,
                                                          true);
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
  static_cast<Derived *>(this)->update_summary_statistics(datum, covariate,
                                                          false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
  if (update_params) {
    static_cast<Derived *>(this)->save_posterior_hypers();
  }
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
bool BaseHierarchy<Derived, State, Hyperparams, Prior>::is_dependent() const {
  return static_cast<Derived const*>(this)->IS_DEPENDENT;
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::check_prior_is_set()
    const {
  if (prior == nullptr) {
    throw std::invalid_argument("Hierarchy prior was not provided");
  }
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
  static_cast<Derived *>(this)->clear_data();
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
