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

template <class Derived, typename State, typename Hyperparams, typename Prior>
class BaseHierarchy : public AbstractHierarchy {
 protected:
  State state;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  Hyperparams posterior_hypers;
  std::shared_ptr<Prior> prior;

  std::set<int> cluster_data_idx;
  int card = 0;
  double log_card = stan::math::NEGATIVE_INFTY;

  virtual void initialize_state() = 0;
  void create_empty_prior() { prior.reset(new Prior); }

  void set_card(const int card_) {
    card = card_;
    log_card = std::log(card_);
  }

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
  Hyperparams get_posterior_hypers() const { return posterior_hypers; }

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

  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;
  //! Removes a datum and its index from the hierarchy
  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  void check_prior_is_set() const;

  virtual google::protobuf::Message *get_mutable_prior() override {
    if (prior == nullptr) create_empty_prior();

    return prior.get();
  }

  int get_card() const override { return card; }
  double get_log_card() const override { return log_card; }
  std::set<int> get_data_idx() const override { return cluster_data_idx; }

  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;
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
      lpdf(i) = static_cast<Derived const *>(this)->like_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else if (covariates.rows() == 1) {
    // Use unique covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->like_lpdf(
          data.row(i), covariates.row(0));
    }
  } else {
    // Use different covariates
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->like_lpdf(
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
