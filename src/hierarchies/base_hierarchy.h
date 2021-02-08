#ifndef BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "hierarchy_id.pb.h"
#include "marginal_state.pb.h"
#include "src/utils/rng.h"

//! Abstract base template class for a hierarchy object.

//! This template class represents a hierarchy object in a generic iterative
//! BNP algorithm, that is, a single set of unique values with their own prior
//! distribution attached to it. These values are part of the Markov chain's
//! state chain (which includes multiple hierarchies) and are simply referred
//! to as the state of the hierarchy. This object also corresponds to a single
//! cluster in the algorithm, in the sense that its state is the set of
//! parameters for the distribution of the data points that belong to it. Since
//! the prior distribution for the state is often the same across multiple
//! different hierarchies, the hyperparameters object is accessed via a shared
//! pointer. Lastly, any hierarchy that inherits from this class contains
//! multiple ways of updating the state, either via prior or posterior
//! distributions, and of evaluating the distribution of the data, either its
//! likelihood (whose parameters are the state) or its marginal distribution.

class AbstractHierarchy {
 protected:
  std::set<int> cluster_data_idx;
  int card = 0;
  double log_card = stan::math::NEGATIVE_INFTY;
  std::shared_ptr<google::protobuf::Message> prior;
  virtual void create_empty_prior() = 0;

 public:
  virtual ~AbstractHierarchy() = default;
  virtual std::shared_ptr<AbstractHierarchy> clone() const = 0;
  virtual bayesmix::HierarchyId get_id() const = 0;

  //! Adds a datum and its index to the hierarchy
  virtual void add_datum(
      const int id, const Eigen::VectorXd &datum,
      const bool update_params = false,
      const Eigen::VectorXd &covariate = Eigen::VectorXd(0)) = 0;
  //! Removes a datum and its index from the hierarchy
  virtual void remove_datum(
      const int id, const Eigen::VectorXd &datum,
      const bool update_params = false,
      const Eigen::VectorXd &covariate = Eigen::VectorXd(0)) = 0;
  //! Deletes all data in the hierarchy
  virtual void initialize() = 0;

  virtual bool is_multivariate() const = 0;
  virtual bool is_dependent() const { return false; }
  virtual bool is_conjugate() const { return true; }
  //!
  virtual void update_hypers(
      const std::vector<bayesmix::MarginalState::ClusterState> &states) = 0;

  // EVALUATION FUNCTIONS FOR SINGLE POINTS
  //! Evaluates the log-likelihood of data in a single point
  virtual double like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) = 0;

  //! Evaluates the log-marginal distribution of data in a single point
  virtual double prior_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) = 0;

  virtual double conditional_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) = 0;

  // EVALUATION FUNCTIONS FOR GRIDS OF POINTS
  //! Evaluates the log-likelihood of data in a grid of points
  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;
  //! Evaluates the log-marginal of data in a grid of points
  virtual Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;

  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  virtual void sample_prior() = 0;
  //! Generates new values for state from the centering posterior distribution
  virtual void sample_full_cond(bool update_params=false) = 0;
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  virtual void check_prior_is_set() = 0;

  // GETTERS AND SETTERS
  int get_card() const { return card; }
  double get_log_card() const { return log_card; }
  std::set<int> get_data_idx() { return cluster_data_idx; }
  google::protobuf::Message *get_mutable_prior() {
    if (prior == nullptr) create_empty_prior();

    return prior.get();
  }
  //! Overloaded version of sample_full_cond(), mainly used for debugging
  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;
};

template <class Derived, typename State, typename Hyperparams, typename Prior>
class BaseHierarchy : public AbstractHierarchy {
 protected:
  State state;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  Hyperparams posterior_hypers;

  virtual Hyperparams get_posterior_parameters() = 0;
  void create_empty_prior() override { prior.reset(new Prior); }
  virtual void initialize_state() = 0;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~BaseHierarchy() = default;
  BaseHierarchy() = default;
  virtual std::shared_ptr<AbstractHierarchy> clone() const override {
    auto out = std::make_shared<Derived>(static_cast<Derived const &>(*this));
    out->clear_data();
    return out;
  }

  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }

  void sample_prior() override {
    state = static_cast<Derived *>(this)->draw(*hypers);
  };
  //! Generates new values for state from the centering posterior distribution
  void sample_full_cond(bool update_params = true) {
    Hyperparams params =
        update_params
            ? static_cast<Derived *>(this)->get_posterior_parameters()
            : posterior_hypers;
    state = static_cast<Derived *>(this)->draw(params);
  }

  void initialize() override {
    hypers = std::make_shared<Hyperparams>();
    check_prior_is_set();
    static_cast<Derived *>(this)->initialize_hypers();
    static_cast<Derived *>(this)->initialize_state();
    posterior_hypers = *hypers;
    static_cast<Derived *>(this)->clear_data();
    }

  void save_posterior_hypers() {
    posterior_hypers =
        static_cast<Derived *>(this)->get_posterior_parameters();
  }

  void add_datum(
      const int id, const Eigen::VectorXd &datum,
      const bool update_params = false,
      const Eigen::VectorXd &covariate = Eigen::VectorXd(0)) override;
  //! Removes a datum and its index from the hierarchy
  void remove_datum(
      const int id, const Eigen::VectorXd &datum,
      const bool update_params = false,
      const Eigen::VectorXd &covariate = Eigen::VectorXd(0)) override;

  void check_prior_is_set();

  double prior_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) {
    return static_cast<Derived *>(this)->marg_lpdf(*hypers, datum, covariate);
  }

  double conditional_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) {
    return static_cast<Derived *>(this)->marg_lpdf(posterior_hypers, datum,
                                                   covariate);
  }

  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

  //! Evaluates the log-marginal of data in a grid of points
  virtual Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;
};

template <class Derived, typename State, typename Hyperparams, typename Prior>
void BaseHierarchy<Derived, State, Hyperparams, Prior>::add_datum(
    const int id, const Eigen::VectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::VectorXd &covariate /*= Eigen::VectorXd(0)*/) {
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
    const int id, const Eigen::VectorXd &datum,
    const bool update_params /*= false*/,
    const Eigen::VectorXd &covariate /* = Eigen::VectorXd(0)*/) {
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
void BaseHierarchy<Derived, State, Hyperparams, Prior>::check_prior_is_set() {
  if (prior == nullptr) {
    throw std::invalid_argument("Hierarchy prior was not provided");
  }
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
BaseHierarchy<Derived, State, Hyperparams, Prior>::like_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = static_cast<Derived *>(this)->like_lpdf(data.row(i),
                                                        Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived *>(this)->like_lpdf(data.row(i),
                                                        covariates.row(i));
    }
  }
  return lpdf;
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
BaseHierarchy<Derived, State, Hyperparams, Prior>::prior_pred_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = static_cast<Derived *>(this)->prior_pred_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived *>(this)->prior_pred_lpdf(
          data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
BaseHierarchy<Derived, State, Hyperparams, Prior>::conditional_pred_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = static_cast<Derived *>(this)->conditional_pred_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived *>(this)->conditional_pred_lpdf(
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
  if (covariates == Eigen::MatrixXd(0, 0)) {
    for (int i = 0; i < data.rows(); i++)
      static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                              Eigen::RowVectorXd(0));
  } else {
    for (int i = 0; i < data.rows(); i++)
      static_cast<Derived *>(this)->add_datum(i, data.row(i), false,
                                              covariates.row(i));
  }
  static_cast<Derived *>(this)->sample_full_cond();
}

#endif  // BAYESMIX_HIERARCHIES_BASE_HIERARCHY_H_
