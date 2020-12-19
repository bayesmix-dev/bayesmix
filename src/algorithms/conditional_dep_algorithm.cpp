#include "conditional_dep_algorithm.hpp"

void ConditionalDepAlgorithm::add_datum_to_hierarchy(BaseHierarchy *hier,
                                                     const int idx) {
  auto *hier_cast = dynamic_cast<BaseDependentHierarchy *>(hier);
  hier_cast->add_datum(idx, &data.row[idx], &covariates.row[idx]);
}

void ConditionalDepAlgorithm::initialize() {
  BaseAlgorithm::initialize();

  // Covariates checks
  assert(data.rows() == covariates.rows());
  // TODO other checks?
}
