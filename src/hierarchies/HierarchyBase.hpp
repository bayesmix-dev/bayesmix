#ifndef HIERARCHYBASE_HPP
#define HIERARCHYBASE_HPP

#include <google/protobuf/message.h>

#include <array>
#include <memory>
#include <random>
#include <stan/math/prim/fun.hpp>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "../utils/rng.hpp"

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

class HierarchyBase {
 protected:
  // AUXILIARY TOOLS
  //! Raises error if the hypers values are not valid w.r.t. their own domain
  virtual void check_hypers_validity() = 0;
  //! Raises error if the state values are not valid w.r.t. their own domain
  virtual void check_state_validity() = 0;

 public:
  virtual void check_and_initialize() = 0;
  //! Returns true if the hierarchy models multivariate data
  virtual bool is_multivariate() const = 0;

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~HierarchyBase() = default;
  HierarchyBase() = default;
  virtual std::shared_ptr<HierarchyBase> clone() const = 0;

  // EVALUATION FUNCTIONS
  //! Evaluates the likelihood of data in the given points
  virtual Eigen::VectorXd like(const Eigen::MatrixXd &data) = 0;

  //! Evaluates the log-likelihood of data in the given points
  virtual Eigen::VectorXd lpdf(const Eigen::MatrixXd &data) = 0;

  //! Evaluates the marginal distribution of data in the given points
  virtual Eigen::VectorXd eval_marg(const Eigen::MatrixXd &data) = 0;

  //! Evaluates the log-marginal distribution of data in the given points
  virtual Eigen::VectorXd marg_lpdf(const Eigen::MatrixXd &data) = 0;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  virtual void draw() = 0;
  //! Generates new values for state from the centering posterior distribution
  virtual void sample_given_data(const Eigen::MatrixXd &data) = 0;

  // GETTERS AND SETTERS
  virtual void get_state_as_proto(google::protobuf::Message *out) = 0;

  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  virtual void set_state(google::protobuf::Message *curr, bool check) = 0;

  virtual void print_id() const = 0;  // TODO
};

#endif  // HIERARCHYBASE_HPP
