#ifndef BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_HPP_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include <stan/math/prim.hpp>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"
#include "base_hierarchy.hpp"

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

class DependentHierarchy : public BaseHierarchy {
 public:
  void add_datum(const int id, const Eigen::VectorXd &datum,
    const Eigen::VectorXd &covariate) override {  // TODO with covariates
    auto it = cluster_data_idx.find(id);
    assert(it == cluster_data_idx.end());
    card += 1;
    log_card = std::log(card);
    update_summary_statistics(datum, true);
    cluster_data_idx.insert(id);
  }

  void remove_datum(const int id, const Eigen::VectorXd &datum,
    const Eigen::VectorXd &covariate) override {  // TODO with covariates
    update_summary_statistics(datum, false);
    card -= 1;
    log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
    auto it = cluster_data_idx.find(id);
    assert(it != cluster_data_idx.end());
    cluster_data_idx.erase(it);
  }

  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~DependentHierarchy() = default;
  DependentHierarchy() = default;

  // EVALUATION FUNCTIONS  // TODO
  //! Evaluates the log-likelihood of data in a single point
  virtual double like_lpdf(const Eigen::RowVectorXd &datum) const = 0;
  //! Evaluates the log-likelihood of data in the given points
  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data) const = 0;
  //! Evaluates the log-marginal distribution of data in a single point
  virtual double marg_lpdf(const Eigen::RowVectorXd &datum) const = 0;
  //! Evaluates the log-marginal distribution of data in the given points
  virtual Eigen::VectorXd marg_lpdf_grid(
      const Eigen::MatrixXd &data) const = 0;
};

#endif  // BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_HPP_
