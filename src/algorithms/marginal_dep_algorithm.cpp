#include "marginal_dep_algorithm.hpp"

#include <Eigen/Dense>
#include <cassert>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"

void MarginalDepAlgorithm::initialize() {
  // Set covariates dimension
  assert(unique_values.size() != 0 && "Error: hierarchy was not provided");
  assert(data.rows() == covariates.rows());
  unique_values[0]->set_dim(covariates.cols());

  BaseAlgorithm::initialize();
  // TODO anything else?
}

Eigen::MatrixXd MarginalDepAlgorithm::eval_lpdf(
    const Eigen::MatrixXd &grid,
    BaseCollector<bayesmix::MarginalState> *const coll) {
  return Eigen::MatrixXd::Zero(0, 0);  // TODO
}
