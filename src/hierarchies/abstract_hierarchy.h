#ifndef BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_

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
  virtual void sample_full_cond(bool update_params = false) = 0;
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;
  virtual void write_hypers_to_proto(google::protobuf::Message *out) const = 0;
  virtual void set_state_from_proto(
      const google::protobuf::Message &state_) = 0;

  virtual void check_prior_is_set() = 0;

  // GETTERS AND SETTERS
  virtual int get_card() const = 0;
  virtual double get_log_card() const = 0;
  virtual std::set<int> get_data_idx() = 0;
  virtual google::protobuf::Message *get_mutable_prior() = 0;

  //! Overloaded version of sample_full_cond(), mainly used for debugging
  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) = 0;
};

#endif  // BAYESMIX_HIERARCHIES_ABSTRACT_HIERARCHY_H_
