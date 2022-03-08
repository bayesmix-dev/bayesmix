#ifndef BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"
#include "src/hierarchies/updaters/abstract_updater.h"
#include "src/utils/rng.h"

//! Abstract base class for a hierarchy object.
//! This class is the basis for a curiously recurring template pattern (CRTP)
//! for `Hierarchy` objects, and is solely composed of interface functions for
//! derived classes to use. For more information about this pattern, as well
//! the list of methods required for classes in this inheritance tree, please
//! refer to the README.md file included in this folder.

//! This abstract class represents a Bayesian hierarchical model:
//! x_1, ..., x_n \sim f(x | \theta)
//!         theta \sim G
//! A Hierarchy object can compute the following quantities:
//! 1- the likelihood log-probability density function
//! 2- the prior predictive probability: \int_\Theta f(x | theta) G(d\theta)
//!    (for conjugate models only)
//! 3- the posterior predictive probability
//!    \int_\Theta f(x | theta) G(d\theta | x_1, ..., x_n)
//!    (for conjugate models only)
//! Moreover, the Hierarchy knows how to sample from the full conditional of
//! theta, possibly in an approximate way.
//!
//! In the context of our Gibbs samplers, an hierarchy represents the parameter
//! value associated to a certain cluster, and also knows which observations
//! are allocated to that cluster.
//! Moreover, hyperparameters and (possibly) hyperpriors associated to them can
//! be shared across multiple Hierarchies objects via a shared pointer.
//! In conjunction with a single `Mixing` object, a collection of `Hierarchy`
//! objects completely defines a mixture model, and these two parts can be
//! chosen independently of each other.
//! Communication with other classes, as well as storage of some relevant
//! values, is performed via appropriately defined Protobuf messages (see for
//! instance the proto/ls_state.proto and proto/hierarchy_prior.proto files)
//! and their relative class methods.

class AbstractHierarchy {
 public:
  //! Set the likelihood for the current hierarchy. Implemented in the
  //! BaseHierarchy class
  virtual void set_likelihood(std::shared_ptr<AbstractLikelihood> like_) = 0;

  //! Set the prior model for the current hierarchy. Implemented in the
  //! BaseHierarchy class
  virtual void set_prior(std::shared_ptr<AbstractPriorModel> prior_) = 0;

  //! Set the update algorithm for the current hierarchy. Implemented in the
  //! BaseHierarchy class
  virtual void set_updater(std::shared_ptr<AbstractUpdater> updater_) = 0;

  //! Returns (a pointer to) the likelihood for the current hierarchy.
  //! Implemented in the BaseHierarchy class
  virtual std::shared_ptr<AbstractLikelihood> get_likelihood() = 0;

  //! Returns (a pointer to) the prior model for the current hierarchy.
  //! Implemented in the BaseHierarchy class
  virtual std::shared_ptr<AbstractPriorModel> get_prior() = 0;

  //! Default destructor
  virtual ~AbstractHierarchy() = default;

  //! Returns an independent, data-less copy of this object
  virtual std::shared_ptr<AbstractHierarchy> clone() const = 0;

  //! Returns an independent, data-less copy of this object
  virtual std::shared_ptr<AbstractHierarchy> deep_clone() const = 0;

  // EVALUATION FUNCTIONS FOR SINGLE POINTS
  //! Public wrapper for `like_lpdf()` methods
  virtual double get_like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (is_dependent()) {
      return like_lpdf(datum, covariate);
    } else {
      return like_lpdf(datum);
    }
  }

  //! Evaluates the log-prior predictive distribution of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double prior_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (is_conjugate()) {
      throw std::runtime_error(
          "prior_pred_lpdf() not implemented for this hierarchy");
    } else {
      throw std::runtime_error(
          "Cannot call prior_pred_lpdf() from a non-conjugate hierarchy");
    }
  }

  //! Evaluates the log-conditional predictive distr. of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double conditional_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    if (is_conjugate()) {
      throw std::runtime_error(
          "conditional_pred_lpdf() not implemented for this hierarchy");
    } else {
      throw std::runtime_error(
          "Cannot call conditional_pred_lpdf() from a non-conjugate "
          "hierarchy");
    }
  }

  // EVALUATION FUNCTIONS FOR GRIDS OF POINTS
  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) const = 0;

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd prior_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) const {
    if (is_conjugate()) {
      throw std::runtime_error(
          "prior_pred_lpdf_grid() not implemented for this hierarchy");
    } else {
      throw std::runtime_error(
          "Cannot call prior_pred_lpdf_grid() from a non-conjugate hierarchy");
    }
  }

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) const {
    if (is_conjugate()) {
      throw std::runtime_error(
          "conditional_pred_lpdf_grid() not implemented for this hierarchy");
    } else {
      throw std::runtime_error(
          "Cannot call conditional_pred_lpdf_grid() from a non-conjugate "
          "hierarchy");
    }
  }

  // SAMPLING FUNCTIONS
  //! Generates new state values from the centering prior distribution
  virtual void sample_prior() = 0;

  //! Generates new state values from the centering posterior distribution
  //! @param update_params  Save posterior hypers after the computation?
  virtual void sample_full_cond(const bool update_params = false) = 0;

  //! Overloaded version of sample_full_cond(bool), mainly used for debugging
  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;

  //! Updates hyperparameter values given a vector of cluster states
  virtual void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states) = 0;

  // GETTERS AND SETTERS
  //! Returns the Protobuf ID associated to this class
  virtual bayesmix::HierarchyId get_id() const = 0;

  //! Returns the current cardinality of the cluster
  virtual int get_card() const = 0;

  //! Returns the logarithm of the current cardinality of the cluster
  virtual double get_log_card() const = 0;

  //! Returns the indexes of data points belonging to this cluster
  virtual std::set<int> get_data_idx() const = 0;

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  virtual google::protobuf::Message *get_mutable_prior() = 0;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! AlgoritmState::ClusterState message by adding the appropriate type
  virtual std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
  get_state_proto() const = 0;

  //! Writes current state to a Protobuf message by pointer
  virtual void write_state_to_proto(
      google::protobuf::Message *const out) const = 0;

  //! Writes current hyperparameters to a Protobuf message by pointer
  virtual void write_hypers_to_proto(
      google::protobuf::Message *const out) const = 0;

  //! Read and set state values from a given Protobuf message
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  //! Read and set hyperparameter values from a given Protobuf message
  virtual void set_hypers_from_proto(
      const google::protobuf::Message &state_) = 0;

  // DATA FUNCTIONS
  //! Adds a datum and its index to the hierarchy
  virtual void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) = 0;

  //! Removes a datum and its index from the hierarchy
  virtual void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) = 0;

  //! Public wrapper for `update_summary_statistics()` methods
  void update_ss(const Eigen::RowVectorXd &datum,
                 const Eigen::RowVectorXd &covariate, const bool add) {
    if (is_dependent()) {
      return update_summary_statistics(datum, covariate, add);
    } else {
      return update_summary_statistics(datum, add);
    }
  }

  // INITIALIZATION FUNCTIONS
  //! Main function that initializes members to appropriate values
  virtual void initialize() = 0;

  // HIERARCHY FEATURES
  //! Returns whether the hierarchy models multivariate data or not
  virtual bool is_multivariate() const = 0;

  //! Returns whether the hierarchy depends on covariate values or not
  virtual bool is_dependent() const = 0;

  //! Returns whether the hierarchy represents a conjugate model or not
  virtual bool is_conjugate() const = 0;

  //! Main function that initializes members to appropriate values
  virtual void set_dataset(const Eigen::MatrixXd *const dataset) = 0;

 protected:
  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  Covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double like_lpdf(const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call like_lpdf() from a non-dependent hierarchy");
    } else {
      throw std::runtime_error(
          "like_lpdf() not implemented for this hierarchy");
    }
  }

  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  virtual double like_lpdf(const Eigen::RowVectorXd &datum) const {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call like_lpdf() from a dependent hierarchy");
    } else {
      throw std::runtime_error(
          "like_lpdf() not implemented for this hierarchy");
    }
  }

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param covariate  Covariate vector associated to datum
  //! @param add        Whether the datum is being added or removed
  virtual void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                         const Eigen::RowVectorXd &covariate,
                                         const bool add) {
    if (!is_dependent()) {
      throw std::runtime_error(
          "Cannot call update_summary_statistics() from a non-dependent "
          "hierarchy");
    } else {
      throw std::runtime_error(
          "update_summary_statistics() not implemented for this hierarchy");
    }
  }

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  virtual void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                         const bool add) {
    if (is_dependent()) {
      throw std::runtime_error(
          "Cannot call update_summary_statistics() from a dependent "
          "hierarchy");
    } else {
      throw std::runtime_error(
          "update_summary_statistics() not implemented for this hierarchy");
    }
  }
};

#endif  // BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_
