#include "marginal_algorithm.h"

#include <Eigen/Dense>
#include <cassert>
#include <stan/math/prim/fun.hpp>

#include "algorithm_state.pb.h"
#include "base_algorithm.h"
#include "src/collectors/base_collector.h"

void MarginalAlgorithm::remove_singleton(const unsigned int idx) {
  // Relabel allocations
  for (auto &c : allocations) {
    if (c > idx) {
      c -= 1;
    }
  }
  // Remove cluster
  unique_values.erase(unique_values.begin() + idx);
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
    const Eigen::RowVectorXd &mix_covariate) {
  // Read mixing state
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  mixing->set_state_from_proto(curr_state.mixing_state());
  // Initialize estimate containers
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
  Eigen::VectorXd lpdf_final(grid.rows());
  auto temp_hier = unique_values[0]->clone();
  temp_hier->set_hypers_from_proto(curr_state.hierarchy_hypers());

  for (size_t j = 0; j < n_clust; j++) {
    // Get hierarchy and mass values
    temp_hier->set_state_from_proto(curr_state.cluster_states(j));
    double mass_ex = mixing->get_mass_existing_cluster(
        n_data, true, false, temp_hier, n_clust, mix_covariate);
    // Get local, single-point estimate
    lpdf_local.col(j) =
        temp_hier->like_lpdf_grid(grid, hier_covariate).array() + mass_ex;
  }
  double mass_new = mixing->get_mass_new_cluster(n_data, true, false, n_clust,
                                                 mix_covariate);
  lpdf_local.col(n_clust) =
      lpdf_marginal_component(temp_hier, grid, hier_covariate).array() +
      mass_new;
  // Loop over grid points
  for (size_t i = 0; i < grid.rows(); i++) {
    // Final estimate for i-th grid point
    lpdf_final(i) = stan::math::log_sum_exp(lpdf_local.row(i));
    assert(!isnan(lpdf_final(i)));
  }
  return lpdf_final;
}
