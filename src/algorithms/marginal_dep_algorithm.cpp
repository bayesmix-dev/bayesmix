#include "marginal_dep_algorithm.hpp"

void MarginalDepAlgorithm::initialize() {
  BaseAlgorithm::initialize();

  // Covariates checks
  assert(data.rows() == covariates.rows());
  // TODO other checks?
}
