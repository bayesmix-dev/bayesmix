#ifndef BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
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
  virtual ~AbstractHierarchy() = default;

  //! Returns an independent copy of this object
  virtual std::shared_ptr<AbstractHierarchy> clone() const = 0;

  // EVALUATION FUNCTIONS FOR SINGLE POINTS
  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const = 0;

  //! Evaluates the log-prior predictive distribution of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double prior_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    throw std::runtime_error(
        "Cannot call prior_pred_lpdf() from a non-conjugate hieararchy");
  }

  //! Evaluates the log-conditional predictive distr. of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double conditional_pred_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    throw std::runtime_error(
        "Cannot call conditional_pred_lpdf() from a non-conjugate hieararchy");
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
    throw std::runtime_error(
        "Cannot call prior_pred_lpdf_grid() from a non-conjugate hieararchy");
  }

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) const {
    throw std::runtime_error(
        "Cannot call conditional_pred_lpdf_grid() from a non-conjugate "
        "hieararchy");
  }

  // SAMPLING FUNCTIONS
  //! Generates new state values from the centering prior distribution
  virtual void sample_prior() = 0;

  //! Generates new state values from the centering posterior distribution
  //! @param update_params  Save posterior hypers after the computation?
  virtual void sample_full_cond(bool update_params = false) = 0;

  //! Updates hyperparameter values given a vector of cluster states
  virtual void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states) = 0;

  // GETTERS AND SETTERS
  //! Returns the current cardinality of the cluster
  virtual int get_card() const = 0;

  //! Returns the logarithm of the current cardinality of the cluster
  virtual double get_log_card() const = 0;

  //! Returns the indexes of data points belonging to this cluster
  virtual std::set<int> get_data_idx() const = 0;

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  virtual google::protobuf::Message *get_mutable_prior() = 0;

  //! Write current state to a Protobuf message by pointer
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;

  //! Write current state to a Protobuf message and return a unique_ptr
  virtual std::unique_ptr<google::protobuf::Message> get_state_proto()
      const = 0;

  //! Write current hyperparameters to a Protobuf message by pointer
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;

  //! Read and set state values from a given Protobuf message
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  // MISCELLANEOUS
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

  //! Initializes class members to appropriate values
  virtual void initialize() = 0;

  //! Returns whether the hierarchy models multivariate data or not
  virtual bool is_multivariate() const = 0;

  //! Returns whether the hierarchy depends on covariate values or not
  virtual bool is_dependent() const { return false; }

  //! Returns whether the hierarchy represents a conjugate model or not
  virtual bool is_conjugate() const { return false; }

  //! Returns the Protobuf ID associated to this class
  virtual bayesmix::HierarchyId get_id() const = 0;

  //! Returns the name of the protobuf message that stores the state
  virtual std::string proto_state_type() const = 0;

  //! Overloaded version of sample_full_cond(), mainly used for debugging
  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;
};

#endif  // BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_
