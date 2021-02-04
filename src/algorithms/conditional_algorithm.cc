#include "conditional_algorithm.h"

#include <Eigen/Dense>

#include "src/collectors/base_collector.h"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
Eigen::MatrixXd ConditionalAlgorithm::eval_lpdf(
    BaseCollector *const collector, const Eigen::MatrixXd &grid,
    const Eigen::MatrixXd &hier_covariates /*= Eigen::MatrixXd(0, 0)*/,
    const Eigen::MatrixXd &mix_covariates /*= Eigen::MatrixXd(0, 0)*/) const {
  return Eigen::MatrixXd::Zero(1, 1);
}
