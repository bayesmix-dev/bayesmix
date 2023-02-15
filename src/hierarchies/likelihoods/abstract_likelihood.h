#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_ABSTRACT_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_ABSTRACT_LIKELIHOOD_H_

#include <google/protobuf/message.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"

//! Abstract class for a generic likelihood
//!
//! This class is the basis for a curiously recurring template pattern (CRTP)
//! for `Likelihood` objects, and is solely composed of interface functions for
//! derived classes to use.
//!
//! A likelihood can evaluate the log probability density faction (lpdf) at a
//! certain point given the current value of the parameters, or compute
//! directly the lpdf for the whole cluster.
//!
//! Whenever possible, we store in a `Likelihood` instance also the sufficient
//! statistics of the data allocated to the cluster, in order to speed-up
//! computations.

class AbstractLikelihood {
 public:
  //! Default destructor
  virtual ~AbstractLikelihood() = default;

  //! Returns an independent, data-less copy of this object
  virtual std::shared_ptr<AbstractLikelihood> clone() const = 0;

  //! Public wrapper for `compute_lpdf()` methods
  double lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (is_dependent() and covariate.size() != 0) {
      return compute_lpdf(datum, covariate);
    } else {
      return compute_lpdf(datum);
    }
  }

  //! Evaluates the log likelihood over all the data in the cluster
  //! given unconstrained parameter values.
  //! By unconstrained parameters we mean that each entry of
  //! the parameter vector can range over (-inf, inf).
  //! Usually, some kind of transformation is required from the unconstrained
  //! parameterization to the actual parameterization.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood over all data in the cluster
  virtual double cluster_lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) const {
    throw std::runtime_error(
        "cluster_lpdf_from_unconstrained() not implemented for this "
        "likelihood");
  }

  //! This version using `stan::math::var` type is required for Stan automatic
  //! differentiation. Evaluates the log likelihood over all the data in the
  //! cluster given unconstrained parameter values. By unconstrained parameters
  //! we mean that each entry of the parameter vector can range over (-inf,
  //! inf). Usually, some kind of transformation is required from the
  //! unconstrained parameterization to the actual parameterization.
  //! @param unconstrained_params vector collecting the unconstrained
  //! parameters
  //! @return The evaluation of the log likelihood over all data in the cluster
  virtual stan::math::var cluster_lpdf_from_unconstrained(
      Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> unconstrained_params)
      const {
    throw std::runtime_error(
        "cluster_lpdf_from_unconstrained() not implemented for this "
        "likelihood");
  }

  virtual Eigen::VectorXd sample() const = 0;

  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) const = 0;

  //! Returns whether the likelihood models multivariate data or not
  virtual bool is_multivariate() const = 0;

  //! Returns whether the likelihood depends on covariate values or not
  virtual bool is_dependent() const = 0;

  //! Read and set state values from a given Protobuf message
  virtual void set_state_from_proto(const google::protobuf::Message &state_,
                                    bool update_card = true) = 0;

  //! Read and set state values from the vector of unconstrained parameters
  virtual void set_state_from_unconstrained(
      const Eigen::VectorXd &unconstrained_state) = 0;

  //! Writes current state to a Protobuf message by pointer
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;

  //! Sets the (pointer to) the dataset in the cluster
  virtual void set_dataset(const Eigen::MatrixXd *const dataset) = 0;

  //! Adds a datum and its index to the likelihood
  virtual void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) = 0;

  //! Removes a datum and its index from the likelihood
  virtual void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) = 0;

  //! Public wrapper for `update_sum_stats()` methods
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate,
                                 bool add) {
    if (is_dependent()) {
      return update_sum_stats(datum, covariate, add);
    } else {
      return update_sum_stats(datum, add);
    }
  }

  //! Resets the values of the summary statistics in the likelihood
  virtual void clear_summary_statistics() = 0;

  virtual void clear_data() = 0;

  //! Returns the vector of the unconstrained parameters for this likelihood
  virtual Eigen::VectorXd get_unconstrained_state() = 0;

 protected:
  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  virtual double compute_lpdf(const Eigen::RowVectorXd &datum) const {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call compute_lpdf() from a dependent likelihood");
    } else {
      throw std::runtime_error(
          "compute_lpdf() not implemented for this likelihood");
    }
  }

  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  Covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double compute_lpdf(const Eigen::RowVectorXd &datum,
                              const Eigen::RowVectorXd &covariate) const {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call compute_lpdf() from a non-dependent likelihood");
    } else {
      throw std::runtime_error(
          "compute_lpdf() not implemented for this likelihood");
    }
  }

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  virtual void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from a dependent hierarchy");
    } else {
      throw std::runtime_error("Not implemented");
    }
  }

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param covariate  Covariate vector associated to datum
  //! @param add        Whether the datum is being added or removed
  virtual void update_sum_stats(const Eigen::RowVectorXd &datum,
                                const Eigen::RowVectorXd &covariate,
                                bool add) {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call this function from a non-dependent hierarchy");
    } else {
      throw std::runtime_error("Not implemented");
    }
  }
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_ABSTRACT_LIKELIHOOD_H_
