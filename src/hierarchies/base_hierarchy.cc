#include "base_hierarchy.h"

#include <Eigen/Dense>
#include <cassert>
#include <set>
#include <stan/math/prim.hpp>

void AbstractHierarchy::add_datum(const int id, const Eigen::VectorXd &datum) {
  auto it = cluster_data_idx.find(id);
  assert(it == cluster_data_idx.end());
  card += 1;
  log_card = std::log(card);
  update_summary_statistics(datum, true);
  cluster_data_idx.insert(id);
}

void AbstractHierarchy::remove_datum(const int id, const Eigen::VectorXd &datum) {
  update_summary_statistics(datum, false);
  card -= 1;
  log_card = (card == 0) ? stan::math::NEGATIVE_INFTY : std::log(card);
  auto it = cluster_data_idx.find(id);
  assert(it != cluster_data_idx.end());
  cluster_data_idx.erase(it);
}

void AbstractHierarchy::check_prior_is_set() {
  if (prior == nullptr) {
    throw std::invalid_argument("Hierarchy prior was not provided");
  }
}

Eigen::VectorXd AbstractHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = like_lpdf(data.row(i));
  }
  return result;
}

Eigen::VectorXd AbstractHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = marg_lpdf(data.row(i));
  }
  return result;
}
