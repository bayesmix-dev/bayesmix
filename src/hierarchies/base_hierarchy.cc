#include "base_hierarchy.h"

#include <Eigen/Dense>
#include <set>
#include <stan/math/prim.hpp>

void BaseHierarchy::add_datum(const int id, const Eigen::VectorXd &datum) {
  auto it = cluster_data_idx.find(id);
  if (it != cluster_data_idx.end()) {
    std::cout << "Warning: data index already in hierarchy, no action taken"
              << std::endl;
    return;
  }
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
  if (it == cluster_data_idx.end()) {
    throw std::invalid_argument("Datum index was not found in hierarchy");
  }
  cluster_data_idx.erase(it);
}

Eigen::VectorXd BaseHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = like_lpdf(data.row(i));
  }
  return result;
}

Eigen::VectorXd BaseHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = marg_lpdf(data.row(i));
  }
  return result;
}
