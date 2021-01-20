#include "marginal_algorithm.hpp"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "../../lib/progressbar/progressbar.hpp"
#include "../collectors/base_collector.hpp"
#include "../utils/eigen_utils.hpp"
#include "marginal_state.pb.h"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
//! \return     Matrix whose i-th column is the lpdf at i-th iteration
Eigen::MatrixXd MarginalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                             BaseCollector *coll) {
  std::deque<Eigen::VectorXd> lpdf;
  bool keep = true;
  progresscpp::ProgressBar bar(coll->get_size(), 60);

  // Loop over non-burn-in algorithm iterations
  while (keep) {
    keep = update_state_from_collector(coll);
    if (!keep) {
      break;
    }
    lpdf.push_back(lpdf_from_state(grid));
    ++bar;
    bar.display();
  }
  coll->reset();
  bar.done();
  return bayesmix::stack_vectors(lpdf);
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid) {
  Eigen::VectorXd out(grid.rows());
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  mixing->set_state_from_proto(curr_state.mixing_state());

  // Initialize local matrix of log-densities
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
  std::shared_ptr<BaseHierarchy> temp_hier = unique_values[0]->clone();

  Eigen::VectorXd weights(n_clust + 1);
  for (size_t j = 0; j < n_clust; j++) {
    // Extract and copy unique values in temp_hier
    temp_hier->set_state_from_proto(curr_state.cluster_states(j));
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
    out(j) = stan::math::log_sum_exp(lpdf_local.row(j));
  }
  return out;
}

bool MarginalAlgorithm::update_state_from_collector(BaseCollector *coll) {
  bool success = coll->get_next_state(&curr_state);
  return success;
}
