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

class DependentHierarchy : public BaseHierarchy {
 public:
  void add_datum(
      const int id, const Eigen::VectorXd &datum,
      const Eigen::VectorXd &covariate) override {  // TODO with covariates
    auto it = cluster_data_idx.find(id);
    assert(it == cluster_data_idx.end());
    card += 1;
    log_card = std::log(card);
    update_summary_statistics(datum, true);
    cluster_data_idx.insert(id);
  }

  void remove_datum(
      const int id, const Eigen::VectorXd &datum,
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
