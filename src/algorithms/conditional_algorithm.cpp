#include "conditional_algorithm.hpp"

#include <Eigen/Dense>

#include "../collectors/base_collector.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
Eigen::MatrixXd ConditionalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                                BaseCollector *coll) {
  return Eigen::MatrixXd::Zero(1, 1);
}
