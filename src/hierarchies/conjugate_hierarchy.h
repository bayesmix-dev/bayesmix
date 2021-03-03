#ifndef BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_

#include "base_hierarchy.h"

template <class Derived, typename State, typename Hyperparams, typename Prior>
class ConjugateHierarchy
    : public BaseHierarchy<Derived, State, Hyperparams, Prior> {
 public:
  ~ConjugateHierarchy() = default;
  ConjugateHierarchy() = default;
  bool is_conjugate() const { return true; }

  using BaseHierarchy<Derived, State, Hyperparams, Prior>::hypers;
  using BaseHierarchy<Derived, State, Hyperparams, Prior>::posterior_hypers;
  using BaseHierarchy<Derived, State, Hyperparams, Prior>::state;

  void save_posterior_hypers() {
    posterior_hypers =
        static_cast<Derived *>(this)->get_posterior_parameters();
  }

  double prior_pred_lpdf(const Eigen::RowVectorXd &datum,
                         const Eigen::RowVectorXd &covariate =
                             Eigen::VectorXd(0)) const override {
    return static_cast<Derived const *>(this)->marg_lpdf(*hypers, datum,
                                                         covariate);
  }

  double conditional_pred_lpdf(const Eigen::RowVectorXd &datum,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::VectorXd(0)) const override {
    return static_cast<Derived const *>(this)->marg_lpdf(posterior_hypers,
                                                         datum, covariate);
  }

  //! Generates new values for state from the centering posterior distribution
  void sample_full_cond(bool update_params = true) override {
    if (this->card == 0) {
      // No posterior update possible
      static_cast<Derived *>(this)->sample_prior();
    } else {
      Hyperparams params =
          update_params
              ? static_cast<Derived *>(this)->get_posterior_parameters()
              : posterior_hypers;
      state = static_cast<Derived *>(this)->draw(params);
    }
  }

  //! Evaluates the log-marginal of data in a grid of points
  virtual Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;
};

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
ConjugateHierarchy<Derived, State, Hyperparams, Prior>::prior_pred_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
          data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd ConjugateHierarchy<Derived, State, Hyperparams, Prior>::
    conditional_pred_lpdf_grid(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    for (int i = 0; i < data.rows(); i++) {
      // Pass null value as covariate
      lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else {
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
          data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

#endif
