#include "algorithm_conditional.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
Eigen::MatrixXd AlgorithmConditional::eval_lpdf(const Eigen::MatrixXd &grid,
                                                CollectorBase *coll) {
  return Eigen::MatrixXd::Zero(1, 1);
}
