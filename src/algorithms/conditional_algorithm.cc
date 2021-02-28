#include "conditional_algorithm.h"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "algorithm_state.pb.h"
#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/conditional_mixing.h"

void ConditionalAlgorithm::initialize() {
  BaseAlgorithm::initialize();
  cond_mixing = std::dynamic_pointer_cast<ConditionalMixing>(mixing);
}

Eigen::VectorXd ConditionalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::VectorXd &hier_covariate,
    const Eigen::VectorXd &mix_covariate) {
  // Read mixing state
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  cond_mixing->set_state_from_proto(curr_state.mixing_state());
  // Initialize estimate containers
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust);
  Eigen::VectorXd lpdf_final(grid.rows());
  auto temp_hier = unique_values[0]->clone();
  // Loop over grid points
  for (size_t i = 0; i < grid.rows(); i++) {
    // Get mixing weights for the i-th grid point
    Eigen::VectorXd logweights =
        cond_mixing->get_weights(true, false, mix_covariate);
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      temp_hier->set_state_from_proto(curr_state.cluster_states(j));
      // Get local, single-point estimate
      lpdf_local(i, j) =
          logweights(j) + temp_hier->like_lpdf(grid.row(i), hier_covariate);
    }
    // Final estimate for i-th grid point
    lpdf_final(i) = stan::math::log_sum_exp(lpdf_local.row(i));
  }
  return lpdf_final;
}
