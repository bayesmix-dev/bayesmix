#ifndef MARGINALALGORITHM_IMP_HPP
#define MARGINALALGORITHM_IMP_HPP

#include "MarginalAlgorithm.hpp"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
template <template <class> class Hierarchy, class Hypers, class Mixing>
void MarginalAlgorithm<Hierarchy, Hypers, Mixing>::eval_density(
    const Eigen::MatrixXd &grid, BaseCollector *coll) {
  // Initialize objects
  this->density.first = grid;
  Eigen::VectorXd dens(Eigen::MatrixXd::Zero(grid.rows(), 1));

  // Read chain from collector
  std::deque<State> chain = coll->get_chain();
  unsigned n_iter = chain.size();
  unsigned int n = chain[0].allocations_size();
  unsigned n_params = chain[0].uniquevalues(0).params_size();

  std::vector<Eigen::MatrixXd> params(n_params);

  // Loop over non-burn-in algorithm iterations
  for (size_t iter = 0; iter < n_iter; iter++) {
    // Compute clusters cardinalities
    unsigned int n_clust = chain[iter].uniquevalues_size();
    std::vector<unsigned int> card(n_clust, 0);
    for (size_t j = 0; j < n; j++) {
      card[chain[iter].allocations(j)] += 1;
    }
    // Initialize temporary hierarchy
    Hierarchy<Hypers> temp_hier(this->unique_values[0].get_hypers());

    // Loop over current iteration's unique values
    for (size_t h = 0; h < n_clust; h++) {
      // Extract and copy unique values in temp_hier
      for (size_t k = 0; k < n_params; k++) {
        params[k] =
            this->proto_param_to_matrix(chain[iter].uniquevalues(h).params(k));
      }
      temp_hier.set_state(params, false);

      // Update density estimate (cluster component)
      dens += this->mixing.mass_existing_cluster(card[h], n) *
              temp_hier.like(grid);
    }
    // Update density estimate (marginal component)
    dens += this->mixing.mass_new_cluster(n_clust, n) *
            this->density_marginal_component(temp_hier);
  }

  // Average over iterations
  this->density.second = dens / n_iter;
  // Update flag
  this->density_was_computed = true;
}

#endif  // MARGINALALGORITHM_IMP_HPP
