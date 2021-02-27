#include "marginal_algorithm.h"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "algorithm_state.pb.h"
#include "src/algorithms/base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/marginal_mixing.h"

void MarginalAlgorithm::initialize() {
  BaseAlgorithm::initialize();
  marg_mixing = std::dynamic_pointer_cast<MarginalMixing>(mixing);
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::MatrixXd &hier_covariates,
    const Eigen::MatrixXd &mix_covariates) {
  Eigen::VectorXd lpdf(grid.rows());
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  marg_mixing->set_state_from_proto(curr_state.mixing_state());
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
  auto temp_hier = unique_values[0]->clone();
  for (size_t j = 0; j < n_clust; j++) {
    temp_hier->set_state_from_proto(curr_state.cluster_states(j));
    lpdf_local.col(j) =
        marg_mixing->mass_existing_cluster(n_data, true, false, temp_hier) +
        temp_hier->like_lpdf_grid(grid, hier_covariates).array();
    // TODO add mixing covariate
  }
  lpdf_local.col(n_clust) =
      marg_mixing->mass_new_cluster(n_data, true, false, n_clust) +
      lpdf_marginal_component(temp_hier, grid, hier_covariates).array();
  // TODO add mixing covariate

  for (size_t j = 0; j < grid.rows(); j++) {
    lpdf(j) = stan::math::log_sum_exp(lpdf_local.row(j));
  }
  return lpdf;
}

bool MarginalAlgorithm::update_state_from_collector(BaseCollector *coll) {
  bool success = coll->get_next_state(&curr_state);
  return success;
}
