#include "conditional_algorithm.h"

#include <Eigen/Dense>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/conditional_mixing.h"

void ConditionalAlgorithm::initialize() {
  BaseAlgorithm::initialize();
  cond_mixing = std::dynamic_pointer_cast<ConditionalMixing>(mixing);
  weights = Eigen::VectorXd::Ones(init_num_clusters) / init_num_clusters;
}

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
Eigen::MatrixXd ConditionalAlgorithm::eval_lpdf(
    BaseCollector *const collector, const Eigen::MatrixXd &grid,
    const Eigen::MatrixXd &hier_covariates /*= Eigen::MatrixXd(0, 0)*/,
    const Eigen::MatrixXd &mix_covariates /*= Eigen::MatrixXd(0, 0)*/) {
  return Eigen::MatrixXd::Zero(1, 1);  // TODO
}
