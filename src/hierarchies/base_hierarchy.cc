#include "base_hierarchy.h"

#include <Eigen/Dense>
#include <cassert>
#include <set>
#include <stan/math/prim.hpp>

void BaseHierarchy::add_datum(const int id, const Eigen::VectorXd &datum) {
  auto it = cluster_data_idx.find(id);
  assert(it == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  update_summary_statistics(datum, true);
  cluster_data_idx.insert(id);
}

void BaseHierarchy::remove_datum(const int id, const Eigen::VectorXd &datum) {
  update_summary_statistics(datum, false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
}

double BaseHierarchy::get_like_lpdf(
                     const Eigen::RowVectorXd &datum,
                     const Eigen::RowVectorXd &covariate /*= Eigen::MatrixXd(0, 0)*/) const {
  if (is_dependent() and covariate.size() == 0) {
    throw std::invalid_argument("Dependent hierarchy lpdf was not supplied with covariates");
  } 
  else if (is_dependent() == false and covariate.size() > 0) {
    throw std::invalid_argument("Non-dependent hierarchy lpdf was supplied with covariates");
  }
  return like_lpdf(datum, covariate);
}

double BaseHierarchy::get_marg_prior_lpdf(
                     const Eigen::RowVectorXd &datum,
                     const Eigen::RowVectorXd &covariate /*= Eigen::MatrixXd(0, 0)*/) const {
  return 0;  // TODO
  }
double BaseHierarchy::get_marg_post_lpdf(
                     const Eigen::RowVectorXd &datum,
                     const Eigen::RowVectorXd &covariate /*= Eigen::MatrixXd(0, 0)*/) const {
  return 0;  // TODO
  }
