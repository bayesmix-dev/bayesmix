#include "marginal_algorithm.hpp"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
//! \return     Matrix whose i-th column is the lpdf at i-th iteration
Eigen::MatrixXd MarginalAlgorithm::eval_lpdf(
    const Eigen::MatrixXd &grid,
    BaseCollector<bayesmix::MarginalState> *coll) {
  // Read chain from collector
  std::deque<bayesmix::MarginalState> chain = coll->get_chain();
  unsigned int n_data = chain[0].cluster_allocs_size();
  unsigned int n_iter = chain.size();

  // Initialize objects
  Eigen::MatrixXd lpdf(grid.rows(), n_iter);

  // Loop over non-burn-in algorithm iterations
  for (size_t i = 0; i < n_iter; i++) {
    mixing->set_state_from_proto(chain[i].mixing_state());

    unsigned int n_clust = chain[i].cluster_states_size();
    std::vector<unsigned int> card(chain[i].cluster_cards().data(),
                                   chain[i].cluster_cards().data() + n_clust);
    // Initialize local matrix of log-densities
    Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
    // Initialize local temporary hierarchy
    std::shared_ptr<BaseHierarchy> temp_hier = unique_values[0]->clone();

    Eigen::VectorXd weights(n_clust + 1);
    // Loop over local unique values i.e. clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Extract and copy unique values in temp_hier
      bayesmix::MarginalState::ClusterState curr_val =
          chain[i].cluster_states(j);
      temp_hier->set_state_from_proto(curr_val);
      temp_hier->set_card(card[j]);
      // Compute cluster component (vector + scalar * unity vector)
      lpdf_local.col(j) =
          mixing->mass_existing_cluster(temp_hier, n_data, true, false) +
          temp_hier->like_lpdf_grid(grid).array();
    }
    // Compute marginal component (vector + scalar * unity vector)
    lpdf_local.col(n_clust) =
        mixing->mass_new_cluster(n_clust, n_data, true, false) +
        lpdf_marginal_component(temp_hier, grid).array();
    // Update overall density estimate
    for (size_t j = 0; j < grid.rows(); j++) {
      lpdf(j, i) = stan::math::log_sum_exp(lpdf_local.row(j));
    }

  }
  return lpdf;
}
