#include "marginal_algorithm.h"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "lib/progressbar/progressbar.h"
#include "marginal_state.pb.h"
#include "src/collectors/base_collector.h"
#include "src/utils/eigen_utils.h"

//! \param grid      Grid of points in matrix form to evaluate the density on
//! \param collector Collector containing the algorithm chain
//! \return          Matrix whose i-th column is the lpdf at i-th iteration
Eigen::MatrixXd MarginalAlgorithm::eval_lpdf(
    BaseCollector *const collector, const Eigen::MatrixXd &grid,
    const Eigen::MatrixXd &hier_covariates /*= Eigen::MatrixXd(0, 0)*/,
    const Eigen::MatrixXd &mix_covariates /*= Eigen::MatrixXd(0, 0)*/) {
  std::deque<Eigen::VectorXd> lpdf;
  bool keep = true;
  progresscpp::ProgressBar bar(collector->get_size(), 60);
  while (keep) {
    keep = update_state_from_collector(collector);
    if (!keep) {
      break;
    }
    lpdf.push_back(lpdf_from_state(grid, hier_covariates, mix_covariates));
    ++bar;
    bar.display();
  }
  collector->reset();
  bar.done();
  return bayesmix::stack_vectors(lpdf);
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::MatrixXd &hier_covariates,
    const Eigen::MatrixXd &mix_covariates) {
  Eigen::VectorXd lpdf(grid.rows());
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  mixing->set_state_from_proto(curr_state.mixing_state());
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
  auto temp_hier = unique_values[0]->clone();
  Eigen::VectorXd weights(n_clust + 1);

  int ndata_in_clus = 0;
  double prior = 0;

  for (size_t j = 0; j < n_clust; j++) {
    temp_hier->set_state_from_proto(curr_state.cluster_states(j));
    lpdf_local.col(j) =
        mixing->mass_existing_cluster(n_data, true, false, temp_hier) +
        temp_hier->like_lpdf_grid(grid, hier_covariates).array();
    ndata_in_clus += temp_hier->get_card();
    prior += mixing->mass_existing_cluster(n_data, false, false, temp_hier);
    // TODO add mixing covariate
  }
  assert(ndata_in_clus == n_data);
  lpdf_local.col(n_clust) =
      mixing->mass_new_cluster(n_data, true, false, n_clust) +
      lpdf_marginal_component(temp_hier, grid, hier_covariates).array();
  prior += mixing->mass_new_cluster(n_data, false, false, n_clust);
  assert(prior == 1.0);

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
