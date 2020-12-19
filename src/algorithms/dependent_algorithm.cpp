#include "dependent_algorithm.hpp"

void DependentAlgorithhm::add_datum_to_hierarchy(BaseHierarchy *hier,
                                                 const int idx) {
  hier->add_datum(idx, &data.row[idx], &covariates.row[idx]);
}

void DependentAlgorithhm::initialize() {
  BaseAlgorithm::initialize();

  // Covariates checks
  assert(data.rows() == covariates.rows());
  // TODO other checks?
}
