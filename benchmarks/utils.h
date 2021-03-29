#ifndef BAYESMIX_BENCHMARKS_UTILS_H
#define BAYESMIX_BENCHMARKS_UTILS_H

#include "src/includes.h"

std::shared_ptr<AbstractHierarchy> get_multivariate_nnw_hierarchy(int dim);

std::shared_ptr<AbstractHierarchy> get_univariate_nnig_hierarchy();

std::shared_ptr<BaseMixing> get_dirichlet_mixing();

std::shared_ptr<BaseAlgorithm> get_algorithm(const std::string& id, int dim);

#endif