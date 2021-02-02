#include "neal2_algorithm.h"

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>
#include <vector>

#include "marginal_state.pb.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/mixings/base_mixing.h"
#include "src/mixings/dependent_mixing.h"
#include "src/utils/distributions.h"
#include "src/utils/rng.h"

//! \param hier Hierarchy object
//! \return     Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal2Algorithm::lpdf_marginal_component(
    std::shared_ptr<BaseHierarchy> hier, const Eigen::MatrixXd &grid,
    const Eigen::MatrixXd &covariates) {
  return hier->marg_lpdf_grid(false, grid, covariates);
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
  for (size_t j = 0; j < n_clust; j++) {
    // Probability of being assigned to an already existing cluster
    loglpdf(j) = unique_values[j]->like_lpdf(data.row(data_idx),
                                             hier_covariates.row(data_idx));
  }
  // Probability of being assigned to a newly created cluster
  loglpdf(n_clust) = unique_values[0]->marg_lpdf(
      false, data.row(data_idx), hier_covariates.row(data_idx));
  return loglpdf;
}

void Neal2Algorithm::print_startup_message() const {
  std::string msg = "Running Neal2 algorithm with " +
                    unique_values[0]->get_id() + " hierarchies, " +
                    mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

void Neal2Algorithm::initialize() {
  BaseAlgorithm::initialize();
  if (unique_values[0]->is_conjugate() == false) {
    throw std::invalid_argument(
        "This algorithm only supports conjugate hierarchies");
  }
}

void Neal2Algorithm::sample_allocations() {
  // Initialize relevant values
  unsigned int n_data = data.rows();
  auto &rng = bayesmix::Rng::Instance().get();
  // Loop over data points
  for (size_t i = 0; i < n_data; i++) {
    unsigned int n_clust = unique_values.size();
    bool singleton = (unique_values[allocations[i]]->get_card() <= 1);
    // Remove datum from cluster
    unique_values[allocations[i]]->remove_datum(i, data.row(i),
                                                hier_covariates.row(i));
    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas =
        get_cluster_prior_mass(i) + get_cluster_lpdf(i);
    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
    unsigned int c_old = allocations[i];
    if (c_new == n_clust) {
      std::shared_ptr<BaseHierarchy> new_unique = unique_values[0]->clone();
      new_unique->add_datum(i, data.row(i), hier_covariates.row(i));
      // Generate new unique values with posterior sampling
      new_unique->sample_given_data();
      unique_values.push_back(new_unique);
      allocations[i] = unique_values.size() - 1;
    } else {
      allocations[i] = c_new;
      unique_values[c_new]->add_datum(i, data.row(i), hier_covariates.row(i));
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
