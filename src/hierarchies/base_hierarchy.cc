#include "base_hierarchy.h"

#include <Eigen/Dense>
#include <cassert>
#include <set>
#include <stan/math/prim.hpp>

void BaseHierarchy::add_datum(const int id, const Eigen::VectorXd &datum,
                              const Eigen::VectorXd &covariate) {
  auto it = cluster_data_idx.find(id);
  assert(it == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  update_summary_statistics(datum, covariate, true);
  cluster_data_idx.insert(id);
}

void BaseHierarchy::remove_datum(const int id, const Eigen::VectorXd &datum,
                                 const Eigen::VectorXd &covariate) {
  update_summary_statistics(datum, covariate, false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
}

Eigen::VectorXd BaseHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd lpdf(data.rows());
  for (int i = 0; i < data.rows(); i++) {
    lpdf(i) = like_lpdf(data.row(i), covariates.row(i));
  }
  return lpdf;
}

Eigen::VectorXd BaseHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates,
    const bool posterior = false) const {
  Eigen::VectorXd lpdf(data.rows());
  for (int i = 0; i < data.rows(); i++) {
    lpdf(i) = marg_lpdf(data.row(i), covariates.row(i), posterior);
  }
  return lpdf;
}
