#include "marginal_dep_algorithm.hpp"

void MarginalDepAlgorithm::initialize() {
  BaseAlgorithm::initialize();

  // Covariates checks
  assert(data.rows() == covariates.rows());
  // TODO other checks?
}

Eigen::MatrixXd MarginalDepAlgorithm::eval_lpdf(
    const Eigen::MatrixXd &grid,
    BaseCollector<bayesmix::MarginalState> *const coll) {
  return Eigen::MatrixXd::Zero(0, 0);  // TODO
}
