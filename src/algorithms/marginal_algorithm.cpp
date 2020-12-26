#include "marginal_algorithm.hpp"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "../utils/eigen_utils.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
//! \return     Matrix whose i-th column is the lpdf at i-th iteration
Eigen::MatrixXd MarginalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                             BaseCollector *coll) {
 
  std::deque<Eigen::VectorXd> lpdf;
  bool keep = true;

  // Loop over non-burn-in algorithm iterations
  while(keep) {
    Eigen::VectorXd curr_lpdf(grid.rows());
    bayesmix::MarginalState state;
    keep = coll->get_next_state(&state);
    if (! keep) {
      break;
    }

    unsigned int n_data = state.cluster_allocs_size();
    mixing->set_state_from_proto(state.mixing_state());
    unsigned int n_clust = state.cluster_states_size();
   
    // Initialize local matrix of log-densities
    Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
    std::shared_ptr<BaseHierarchy> temp_hier = unique_values[0]->clone();

    Eigen::VectorXd weights(n_clust + 1);
    for (size_t j = 0; j < n_clust; j++) {
      // Extract and copy unique values in temp_hier
      bayesmix::MarginalState::ClusterState curr_val =
          state.cluster_states(j);
      temp_hier->set_state_from_proto(curr_val);
      // Compute cluster component (vector + scalar * unity vector)
      lpdf_local.col(j) =
          mixing->mass_existing_cluster(temp_hier, n_data, true, false) +
          temp_hier->like_lpdf_grid(grid).array();
    }
    // Compute marginal component (vector + scalar * unity vector)
    lpdf_local.col(n_clust) =
        mixing->mass_new_cluster(n_clust, n_data, true, false) +
        lpdf_marginal_component(temp_hier, grid).array();
    for (size_t j = 0; j < grid.rows(); j++) {
      curr_lpdf(j) = stan::math::log_sum_exp(lpdf_local.row(j));
    }
    lpdf.push_back(curr_lpdf);
  }
  return bayesmix::stack_vectors(lpdf);
}
