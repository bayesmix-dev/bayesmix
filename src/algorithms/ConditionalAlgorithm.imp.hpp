#ifndef CONDITIONALALGORITHM_IMP_HPP
#define CONDITIONALALGORITHM_IMP_HPP

#include "ConditionalAlgorithm.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
template <template <class> class Hierarchy, class Hypers, class Mixing>
void ConditionalAlgorithm<Hierarchy, Hypers, Mixing>::eval_density(
    const Eigen::MatrixXd &grid, BaseCollector *coll) {
  std::cout << "TODO" << std::endl;  // TODO
}

#endif  // CONDITIONALALGORITHM_IMP_HPP
