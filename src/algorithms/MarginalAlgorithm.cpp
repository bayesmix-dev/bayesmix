#include "MarginalAlgorithm.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
void MarginalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                  BaseCollector *coll) {
  // Initialize objects
  lpdf.first = grid;
  Eigen::VectorXd lpdf_(grid.rows());

  // Read chain from collector
  std::deque<State> chain = coll->get_chain();
  unsigned n_iter = chain.size();
  unsigned int n = chain[0].allocations_size();
  unsigned n_params = chain[0].uniquevalues(0).params_size();

  std::vector<Eigen::MatrixXd> params(n_params);

  // Loop over non-burn-in algorithm iterations
  for (size_t i = 0; i < n_iter; i++) {
    // Compute local clusters cardinalities (i.e. of the current iteration)
    unsigned int n_clust = chain[i].uniquevalues_size();
    std::vector<unsigned int> card(n_clust, 0);
    for (size_t j = 0; j < n; j++) {
      card[chain[i].allocations(j)] += 1;
    }
    // Initialize local matrix of log-densities
    Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
    // Initialize local temporary hierarchy
    std::shared_ptr<HierarchyBase> temp_hier = unique_values[0];  // TODO hyp

    // Loop over local unique values i.e. clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Extract and copy unique values in temp_hier
      for (size_t k = 0; k < n_params; k++) {
        params[k] = proto_param_to_matrix(chain[i].uniquevalues(j).params(k));
      }
      temp_hier->set_state(params, false);

      // Compute cluster component (vector + scalar * unity vector)
      lpdf_local.col(j) = log(mixing->mass_existing_cluster(card[j], n)) +
                          temp_hier->lpdf(grid).array();
    }
    // Compute marginal component (vector + scalar * unity vector)
    lpdf_local.col(n_clust) = log(mixing->mass_new_cluster(n_clust, n)) +
                              lpdf_marginal_component(temp_hier).array();
    // Update overall density estimate
    for (size_t j = 0; j < grid.rows(); j++) {
      lpdf_(j) = stan::math::log_sum_exp(lpdf_local.row(j));
    }
  }

  // Average over iterations (in log-space)
  lpdf.second = lpdf_.array() - log(n_iter);
  // Update flag
  lpdf_was_computed = true;
}
