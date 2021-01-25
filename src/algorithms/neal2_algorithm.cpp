#include "neal2_algorithm.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <stan/math/prim/fun.hpp>
#include <vector>

#include "../hierarchies/base_hierarchy.hpp"
#include "../hierarchies/dependent_hierarchy.hpp"
#include "../mixings/base_mixing.hpp"
#include "../mixings/dependent_mixing.hpp"
#include "../utils/distributions.hpp"
#include "../utils/rng.hpp"
#include "marginal_state.pb.h"

//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal2Algorithm::lpdf_marginal_component(
    std::shared_ptr<BaseHierarchy> temp_hier, const Eigen::MatrixXd &grid) {
  // Exploit conjugacy of hierarchy
  return temp_hier->marg_lpdf_grid(grid);
}

Eigen::VectorXd Neal2Algorithm::get_cluster_prior_mass(
    const unsigned int data_idx) const {
  unsigned int n_data = data.rows();
  unsigned int n_clust = unique_values.size();
  Eigen::VectorXd logprior(n_clust + 1);
  if (mixing->is_dependent()) {
    auto mixcast = std::dynamic_pointer_cast<DependentMixing>(mixing);
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      logprior(j) = mixcast->mass_existing_cluster(
          unique_values[j], mix_covariates.row(data_idx), n_data - 1, true,
          true);
    }
    // Further update with marginal component
    logprior(n_clust) = mixcast->mass_new_cluster(
        mix_covariates.row(data_idx), n_clust, n_data - 1, true, true);
  } else {
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      logprior(j) = mixing->mass_existing_cluster(unique_values[j], n_data - 1,
                                                  true, true);
    }
    // Further update with marginal component
    logprior(n_clust) =
        mixing->mass_new_cluster(n_clust, n_data - 1, true, true);
  }
  return logprior;
}

Eigen::VectorXd Neal2Algorithm::get_cluster_lpdf(
    const unsigned int data_idx) const {
  unsigned int n_data = data.rows();
  unsigned int n_clust = unique_values.size();
  Eigen::VectorXd loglpdf(n_clust + 1);
  if (unique_values[0]->is_dependent()) {
    auto hiercast =
        std::dynamic_pointer_cast<DependentHierarchy>(unique_values[0]);
    // Update with marginal component
    loglpdf(n_clust) =
        hiercast->marg_lpdf(data.row(data_idx), mix_covariates.row(data_idx));

    for (size_t j = 0; j < n_clust; j++) {
      hiercast =
          std::dynamic_pointer_cast<DependentHierarchy>(unique_values[j]);
      // Probability of being assigned to an already existing cluster
      loglpdf(j) = hiercast->like_lpdf(data.row(data_idx),
                                       mix_covariates.row(data_idx));
    }

  } else {
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      loglpdf(j) = unique_values[j]->like_lpdf(data.row(data_idx));
    }
    // Further update with marginal component
    loglpdf(n_clust) = unique_values[0]->marg_lpdf(data.row(data_idx));
  }
  return loglpdf;
}

void Neal2Algorithm::print_startup_message() const {
  std::string msg = "Running Neal2 algorithm with " +
                    unique_values[0]->get_id() + " hierarchies, " +
                    mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

void Neal2Algorithm::sample_allocations() {
  // Initialize relevant values
  unsigned int n_data = data.rows();
  int ndata_from_hier = 0;
  // #ifdef DEBUG
  for (auto &clus : unique_values) ndata_from_hier += clus->get_card();
  assert(n_data == ndata_from_hier);
  // #endif
  auto &rng = bayesmix::Rng::Instance().get();

  // Loop over data points
  for (size_t i = 0; i < n_data; i++) {
    unsigned int n_clust = unique_values.size();
    bool singleton = (unique_values[allocations[i]]->get_card() <= 1);
    // Remove datum from cluster
    remove_datum_from_hierarchy(i, unique_values[allocations[i]]);
    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas =
        get_cluster_prior_mass(i) + get_cluster_lpdf(i);
    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
    unsigned int c_old = allocations[i];

    if (c_new == n_clust) {
      std::shared_ptr<BaseHierarchy> new_unique = unique_values[0]->clone();
      add_datum_to_hierarchy(i, new_unique);
      // Generate new unique values with posterior sampling
      new_unique->sample_given_data();
      unique_values.push_back(new_unique);
      allocations[i] = unique_values.size() - 1;
    } else {
      allocations[i] = c_new;
      add_datum_to_hierarchy(i, unique_values[c_new]);
    }
    if (singleton) {
      // Relabel allocations so that they are consecutive numbers
      for (auto &c : allocations) {
        if (c > c_old) {
          c -= 1;
        }
      }
      unique_values.erase(unique_values.begin() + c_old);
    }
  }
}

void Neal2Algorithm::sample_unique_values() {
  for (auto &clus : unique_values) clus->sample_given_data();
}
