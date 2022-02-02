#include "neal8_algorithm.h"

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>

#include "algorithm_id.pb.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "mixing_id.pb.h"
#include "neal2_algorithm.h"
#include "src/algorithms/marginal_algorithm.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/utils/distributions.h"

void Neal8Algorithm::read_params_from_proto(
    const bayesmix::AlgorithmParams &params) {
  // Set number of auxiliary blocks in addition to regular parameters
  BaseAlgorithm::read_params_from_proto(params);
  n_aux = params.neal8_n_aux();
}

void Neal8Algorithm::initialize() {
  MarginalAlgorithm::initialize();
  // Create correct amount of auxiliary blocks
  aux_unique_values.clear();
  for (size_t i = 0; i < n_aux; i++) {
    aux_unique_values.push_back(unique_values[0]->clone());
  }
}

void Neal8Algorithm::print_startup_message() const {
  std::string msg = "Running Neal8 algorithm (m=" + std::to_string(n_aux) +
                    " aux. blocks) with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}

void Neal8Algorithm::sample_allocations() {
  // Initialize relevant values
  unsigned int n_data = data.rows();
  auto &rng = bayesmix::Rng::Instance().get();

  // Loop over data points
  for (size_t i = 0; i < n_data; i++) {
    bool singleton = (unique_values[allocations[i]]->get_card() <= 1);
    unsigned int c_old = allocations[i];

    if (singleton) {
      // Save unique value in the first auxiliary block
      bayesmix::AlgorithmState::ClusterState curr_val;
      unique_values[allocations[i]]->write_state_to_proto(&curr_val);
      aux_unique_values[0]->set_state_from_proto(curr_val);
      // Remove datum from cluster
      remove_singleton(c_old);
    } else {
      unique_values[c_old]->remove_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
    }

    unsigned int n_clust = unique_values.size();
    // Draw the unique values in the auxiliary blocks from their prior
    for (size_t j = singleton; j < n_aux; j++) {
      aux_unique_values[j]->sample_prior();
    }
    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas =
        get_cluster_prior_mass(i) + get_cluster_lpdf(i);
    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);

    if (c_new >= n_clust) {
      // datum moves to a new cluster
      // Copy one of the auxiliary block as the new cluster
      std::shared_ptr<AbstractHierarchy> hier_new =
          aux_unique_values[c_new - n_clust]->clone();
      unique_values.push_back(hier_new);
      allocations[i] = n_clust;
      unique_values[n_clust]->add_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
    } else {
      allocations[i] = c_new;
      unique_values[c_new]->add_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
    }
  }
}

Eigen::VectorXd Neal8Algorithm::lpdf_marginal_component(
    std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &covariate) const {
  unsigned int n_grid = grid.rows();
  Eigen::VectorXd lpdf_(n_grid);
  Eigen::MatrixXd lpdf_temp(n_grid, n_aux);
  for (size_t i = 0; i < n_aux; i++) {
    hier->sample_prior();
    lpdf_temp.col(i) = hier->like_lpdf_grid(grid, covariate);
  }
  for (size_t i = 0; i < n_grid; i++) {
    lpdf_(i) = stan::math::log_sum_exp(lpdf_temp.row(i));
  }
  return lpdf_.array() - log(n_aux);
}

Eigen::VectorXd Neal8Algorithm::get_cluster_prior_mass(
    const unsigned int data_idx) const {
  unsigned int n_data = data.rows();
  unsigned int n_clust = unique_values.size();
  Eigen::VectorXd logprior(n_clust + n_aux);
  for (size_t j = 0; j < n_clust; j++) {
    // Probability of being assigned to an already existing cluster
    logprior(j) = mixing->get_mass_existing_cluster(
        n_data - 1, true, true, unique_values[j], n_clust,
        mix_covariates.row(data_idx));
  }
  // Further update with marginal components
  for (size_t j = 0; j < n_aux; j++) {
    logprior(n_clust + j) = mixing->get_mass_new_cluster(
        n_data - 1, true, true, n_clust, mix_covariates.row(data_idx));
  }
  return logprior;
}

Eigen::VectorXd Neal8Algorithm::get_cluster_lpdf(
    const unsigned int data_idx) const {
  unsigned int n_data = data.rows();
  unsigned int n_clust = unique_values.size();
  Eigen::VectorXd loglpdf(n_clust + n_aux);
  for (size_t j = 0; j < n_clust; j++) {
    // Probability of being assigned to an already existing cluster
    loglpdf(j) = unique_values[j]->get_like_lpdf(
        data.row(data_idx), hier_covariates.row(data_idx));
  }
  for (size_t j = 0; j < n_aux; j++) {
    // Probability of being assigned to a newly created cluster
    loglpdf(n_clust + j) =
        aux_unique_values[j]->get_like_lpdf(data.row(data_idx),
                                            hier_covariates.row(data_idx)) -
        log(n_aux);
  }
  return loglpdf;
}
