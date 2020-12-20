#include "conditional_dep_algorithm.hpp"

void ConditionalDepAlgorithm::initialize() {
  ConditionalAlgorithm::initialize();

  // Covariates checks
  assert(data.rows() == covariates.rows());
  // TODO other checks?
}
