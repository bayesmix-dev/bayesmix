#include "conditional_algorithm.h"

#include <Eigen/Dense>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/conditional_mixing.h"

void ConditionalAlgorithm::initialize() {
  BaseAlgorithm::initialize();
  cond_mixing = std::dynamic_pointer_cast<ConditionalMixing>(mixing);
}

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
Eigen::MatrixXd ConditionalAlgorithm::eval_lpdf() {
  // TODO use the MargAlgo one?
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::MatrixXd &hier_covariates,
    const Eigen::MatrixXd &mix_covariates) {
  Eigen::VectorXd lpdf(grid.rows());
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  marg_mixing->set_state_from_proto(curr_state.mixing_state());
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust);
  auto temp_hier = unique_values[0]->clone();
  Eigen::VectorXd weights = ???;  // TODO
  for (size_t j = 0; j < n_clust; j++) {
    temp_hier->set_state_from_proto(curr_state.cluster_states(j));
    lpdf_local.col(j) = std::log(weights(j)) +  // TODO use log flag?
        temp_hier->like_lpdf_grid(grid, hier_covariates).array();
    // TODO add mixing covariate
  }

  for (size_t j = 0; j < grid.rows(); j++) {
    lpdf(j) = stan::math::log_sum_exp(lpdf_local.row(j));
  }
  return lpdf;
}
