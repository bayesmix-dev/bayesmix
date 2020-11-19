#include "marginal_algorithm.hpp"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
//! \return     Matrix whose i-th column is the lpdf at i-th iteration
Eigen::MatrixXd MarginalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                             BaseCollector *coll) {
  std::cout << "Computing log-density..." << std::endl;

  // Read chain from collector
  std::deque<bayesmix::MarginalState> chain = coll->get_chain();
  unsigned int n_data = chain[0].cluster_allocs_size();
  unsigned int n_iter = chain.size();

  // Initialize objects
  Eigen::MatrixXd lpdf(grid.rows(), n_iter);

  // Loop over non-burn-in algorithm iterations
  for (size_t i = 0; i < n_iter; i++) {
    // Compute local clusters cardinalities (i.e. of the current iteration)
    unsigned int n_clust = chain[i].cluster_vals_size();
    std::vector<unsigned int> card(n_clust, 0);
    for (size_t j = 0; j < n_data; j++) {
      card[chain[i].cluster_allocs(j)] += 1;
    }
    // Initialize local matrix of log-densities
    Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
    // Initialize local temporary hierarchy
    std::shared_ptr<BaseHierarchy> temp_hier = unique_values[0]->clone();

    // Loop over local unique values i.e. clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Extract and copy unique values in temp_hier
      bayesmix::MarginalState::ClusterVal curr_val = chain[i].cluster_vals(j);
      temp_hier->set_state(curr_val, false);

      // Compute cluster component (vector + scalar * unity vector)
      lpdf_local.col(j) = log(mixing->mass_existing_cluster(card[j], n_data)) +
                          temp_hier->like_lpdf_grid(grid).array();
    }
    // Compute marginal component (vector + scalar * unity vector)
    lpdf_local.col(n_clust) = log(mixing->mass_new_cluster(n_clust, n_data)) +
                              lpdf_marginal_component(temp_hier, grid).array();
    // Update overall density estimate
    for (size_t j = 0; j < grid.rows(); j++) {
      lpdf(j, i) = stan::math::log_sum_exp(lpdf_local.row(j));
    }
  }

  std::cout << "Done" << std::endl;
  return lpdf;
}
