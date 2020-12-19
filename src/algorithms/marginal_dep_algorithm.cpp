#include "marginal_dep_algorithm.hpp"

void MarginalDepAlgorithhm::add_datum_to_hierarchy(BaseHierarchy *hier,
                                                 const int idx) {
  hier->add_datum(idx, &data.row[idx], &covariates.row[idx]);
}

void MarginalDepAlgorithhm::initialize() {
  BaseAlgorithm::initialize();

  // Covariates checks
  assert(data.rows() == covariates.rows());
  // TODO other checks?
}
