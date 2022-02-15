#ifndef BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_

#include "base_hierarchy.h"

//! Template base class for conjugate hierarchy objects.

//! This class acts as the base class for conjugate models, i.e. ones for which
//! both the prior and posterior distribution have the same form
//! (non-conjugate hierarchies should instead inherit directly from
//! `BaseHierarchy`). This also means that the marginal distribution for the
//! data is available in closed form. For this reason, each class deriving from
//! this one must have a free method with one of the following signatures,
//! based on whether it depends on covariates or not:
//! double marg_lpdf(
//!      const Hyperparams &params, const Eigen::RowVectorXd &datum,
//!      const Eigen::RowVectorXd &covariate) const;
//!  or
//! double marg_lpdf(
//!      const Hyperparams &params, const Eigen::RowVectorXd &datum) const;
//! This returns the evaluation of the marginal distribution on the given data
//! point (and covariate, if any), conditioned on the provided `Hyperparams`
//! object. The latter may contain either prior or posterior values for
//! hyperparameters, depending on where this function is called within the
//! library.
//! For more information, please refer to parent classes `AbstractHierarchy`
//! and `BaseHierarchy`.

template <class Derived, typename State, typename Hyperparams, typename Prior>
class ConjugateHierarchy
    : public BaseHierarchy<Derived, State, Hyperparams, Prior> {
 public:
  using BaseHierarchy<Derived, State, Hyperparams, Prior>::hypers;
  using BaseHierarchy<Derived, State, Hyperparams, Prior>::posterior_hypers;
  using BaseHierarchy<Derived, State, Hyperparams, Prior>::state;

  ConjugateHierarchy() = default;
  ~ConjugateHierarchy() = default;

  //! Public wrapper for `marg_lpdf()` methods
  virtual double get_marg_lpdf(
      const Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const;

  //! Evaluates the log-prior predictive distribution of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double prior_pred_lpdf(const Eigen::RowVectorXd &datum,
                         const Eigen::RowVectorXd &covariate =
                             Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(*hypers, datum, covariate);
  }

  //! Evaluates the log-conditional predictive distr. of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double conditional_pred_lpdf(const Eigen::RowVectorXd &datum,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override {
    return get_marg_lpdf(posterior_hypers, datum, covariate);
  }

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  //! Generates new state values from the centering posterior distribution
  //! @param update_params  Save posterior hypers after the computation?
  void sample_full_cond(const bool update_params = true) override {
    if (this->card == 0) {
      // No posterior update possible
      static_cast<Derived *>(this)->sample_prior();
    } else {
      Hyperparams params =
          update_params
              ? static_cast<Derived *>(this)->compute_posterior_hypers()
              : posterior_hypers;
      state = static_cast<Derived *>(this)->draw(params);
    }
  }

  //! Saves posterior hyperparameters to the corresponding class member
  void save_posterior_hypers() {
    posterior_hypers =
        static_cast<Derived *>(this)->compute_posterior_hypers();
  }

  //! Returns whether the hierarchy represents a conjugate model or not
  bool is_conjugate() const override { return true; }

 protected:
  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @param covariate  Covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double marg_lpdf(const Hyperparams &params,
                           const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const {
    if (!this->is_dependent()) {
      throw std::runtime_error(
          "Cannot call marg_lpdf() from a non-dependent hierarchy");
    } else {
      throw std::runtime_error("marg_lpdf() not implemented");
    }
  }

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  virtual double marg_lpdf(const Hyperparams &params,
                           const Eigen::RowVectorXd &datum) const {
    if (this->is_dependent()) {
      throw std::runtime_error(
          "Cannot call marg_lpdf() from a dependent hierarchy");
    } else {
      throw std::runtime_error("marg_lpdf() not implemented");
    }
  }
};

template <class Derived, typename State, typename Hyperparams, typename Prior>
double ConjugateHierarchy<Derived, State, Hyperparams, Prior>::get_marg_lpdf(
    const Hyperparams &params, const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  if (this->is_dependent()) {
    return marg_lpdf(params, datum, covariate);
  } else {
    return marg_lpdf(params, datum);
  }
}

template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
ConjugateHierarchy<Derived, State, Hyperparams, Prior>::prior_pred_lpdf_grid(
    const Eigen::MatrixXd &data,
    const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  Eigen::VectorXd lpdf(data.rows());
  if (covariates.cols() == 0) {
    // Pass null value as covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else if (covariates.rows() == 1) {
    // Use unique covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->prior_pred_lpdf(
          data.row(i), covariates.row(0));
    }
  } else {
    // Use different covariates
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
    // Pass null value as covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
          data.row(i), Eigen::RowVectorXd(0));
    }
  } else if (covariates.rows() == 1) {
    // Use unique covariate
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
          data.row(i), covariates.row(0));
    }
  } else {
    // Use different covariates
    for (int i = 0; i < data.rows(); i++) {
      lpdf(i) = static_cast<Derived const *>(this)->conditional_pred_lpdf(
          data.row(i), covariates.row(i));
    }
  }
  return lpdf;
}

#endif  // BAYESMIX_HIERARCHIES_CONJUGATE_HIERARCHY_H_
