#include "marginal_algorithm.h"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "lib/progressbar/progressbar.h"
#include "marginal_state.pb.h"
#include "src/collectors/base_collector.h"
#include "src/utils/eigen_utils.h"

//! \param grid Grid of points in matrix form to evaluate the density on
//! \param coll Collector containing the algorithm chain
//! \return     Matrix whose i-th column is the lpdf at i-th iteration
Eigen::MatrixXd MarginalAlgorithm::eval_lpdf(const Eigen::MatrixXd &grid,
                                             const Eigen::MatrixXd &covariates,
                                             BaseCollector *coll) const {
  std::deque<Eigen::VectorXd> lpdf;
  bool keep = true;
  progresscpp::ProgressBar bar(coll->get_size(), 60);
  while (keep) {
    keep = update_state_from_collector(coll);
    if (!keep) {
      break;
    }
    lpdf.push_back(lpdf_from_state(grid, covariates));
    ++bar;
    bar.display();
  }
  coll->reset();
  bar.done();
  return bayesmix::stack_vectors(lpdf);
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::MatrixXd &covariates) {
  Eigen::VectorXd out(grid.rows());
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  mixing->set_state_from_proto(curr_state.mixing_state());
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
  auto temp_hier =
      std::dynamic_pointer_cast<BaseHierarchy>(unique_values[0]->clone());
  Eigen::VectorXd weights(n_clust + 1);
  for (size_t j = 0; j < n_clust; j++) {
    temp_hier->set_state_from_proto(curr_state.cluster_states(j));
    lpdf_local.col(j) =
        mixing->mass_existing_cluster(temp_hier, n_data, true, false) +
        temp_hier->like_lpdf_grid(grid, covariates).array();
  }
  lpdf_local.col(n_clust) =
      mixing->mass_new_cluster(n_clust, n_data, true, false) +
      lpdf_marginal_component(temp_hier, grid, covariates).array();

  for (size_t j = 0; j < grid.rows(); j++) {
    out(j) = stan::math::log_sum_exp(lpdf_local.row(j));
  }
  return out;
}

bool MarginalAlgorithm::update_state_from_collector(BaseCollector *coll) {
  bool success = coll->get_next_state(&curr_state);
  return success;
}
