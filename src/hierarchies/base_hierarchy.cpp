#include "base_hierarchy.hpp"

#include <Eigen/Dense>
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
