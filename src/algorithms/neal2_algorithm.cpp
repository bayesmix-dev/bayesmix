#include "neal2_algorithm.hpp"

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>
#include <vector>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../hierarchies/base_hierarchy.hpp"
#include "../mixings/base_mixing.hpp"
#include "../utils/distributions.hpp"
#include "../utils/rng.hpp"

//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal2Algorithm::lpdf_marginal_component(
    std::shared_ptr<BaseHierarchy> temp_hier, const Eigen::MatrixXd &grid) {
  // Exploit conjugacy of hierarchy
  return temp_hier->marg_lpdf_grid(grid);
}

void Neal2Algorithm::print_startup_message() const {
  std::string msg = "Running Neal2 algorithm with " +
                    unique_values[0]->get_id() + " hierarchies, " +
                    mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

void Neal2Algorithm::initialize() {
  // Initialize objects
  cardinalities.reserve(data.rows());
  std::default_random_engine generator;
  // Build uniform probability on clusters, given their initial number
  std::uniform_int_distribution<int> distro(0, init_num_clusters - 1);

  // Allocate one datum per cluster first, and update cardinalities
  for (size_t i = 0; i < init_num_clusters; i++) {
    allocations.push_back(i);
    cardinalities.push_back(1);
  }

  // Randomly allocate all remaining data, and update cardinalities
  for (size_t i = init_num_clusters; i < data.rows(); i++) {
    unsigned int clust = distro(generator);
    allocations.push_back(clust);
    cardinalities[clust] += 1;
  }
}

void Neal2Algorithm::sample_allocations() {
  // Initialize relevant values
  unsigned int n_data = data.rows();
  auto &rng = bayesmix::Rng::Instance().get();

  // Loop over data points
  for (size_t i = 0; i < n_data; i++) {
    // Current i-th datum as row vector
    Eigen::Matrix<double, 1, Eigen::Dynamic> datum = data.row(i);
    // Initialize current number of clusters
    unsigned int n_clust = unique_values.size();
    // Initialize pseudo-flag
    int singleton = (cardinalities[allocations[i]] == 1) ? 1 : 0;

    // Remove datum from cluster
    cardinalities[allocations[i]] -= 1;

    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas(n_clust + (1 - singleton));
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      logprobas(j) =
          log(mixing->mass_existing_cluster(cardinalities[j], n_data - 1)) +
          unique_values[j]->like_lpdf(datum);
      if (singleton == 1 && j == allocations[i]) {
        // Probability of being assigned to a newly generated cluster
        logprobas(j) = log(mixing->mass_new_cluster(n_clust, n_data - 1)) +
                       unique_values[0]->marg_lpdf(datum);
      }
    }
    if (singleton == 0) {
      // Further update with marginal component
      logprobas(n_clust) = log(mixing->mass_new_cluster(n_clust, n_data - 1)) +
                           unique_values[0]->marg_lpdf(datum);
    }
    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
    // Assign datum to its new cluster and update cardinalities:
    // 4 cases are handled separately
    if (singleton == 1) {
      if (c_new == allocations[i]) {
        // Case 1: datum moves from a singleton to a new cluster
        // Replace former with new cluster by updating unique values
        unique_values[allocations[i]]->sample_given_data(datum);
        cardinalities[c_new] += 1;
      }

      else {  // Case 2: datum moves from a singleton to an old cluster
        unique_values.erase(unique_values.begin() + allocations[i]);
        unsigned int c_old = allocations[i];
        allocations[i] = c_new;
        // Relabel allocations so that they are consecutive numbers
        for (auto &c : allocations) {
          if (c > c_old) {
            c -= 1;
          }
        }
        cardinalities[c_new] += 1;
        cardinalities.erase(cardinalities.begin() + c_old);
      }
    }

    else {  // if singleton == 0
      if (c_new == n_clust) {
        // Case 3: datum moves from a non-singleton to a new cluster
        std::shared_ptr<BaseHierarchy> new_unique = unique_values[0]->clone();
        // Generate new unique values with posterior sampling
        new_unique->sample_given_data(datum);
        unique_values.push_back(new_unique);
        allocations[i] = n_clust;
        cardinalities.push_back(1);
      }

      else {  // Case 4: datum moves from a non-singleton to an old cluster
        allocations[i] = c_new;
        cardinalities[c_new] += 1;
      }
    }
  }
}

void Neal2Algorithm::sample_unique_values() {
  // Initialize relevant values
  unsigned int n_clust = unique_values.size();
  unsigned int n_data = allocations.size();

  // Vector that represents all clusters by the indexes of their data points
  std::vector<std::vector<unsigned int>> clust_idxs(n_clust);
  for (size_t i = 0; i < n_data; i++) {
    clust_idxs[allocations[i]].push_back(i);
  }

  // Loop over clusters
  for (size_t i = 0; i < n_clust; i++) {
    unsigned int curr_size = clust_idxs[i].size();
    // Build vector that contains the data points in the current cluster
    Eigen::MatrixXd curr_data(curr_size, data.cols());
    for (size_t j = 0; j < curr_size; j++) {
      curr_data.row(j) = data.row(clust_idxs[i][j]);
    }
    // Update unique values via the posterior distribution
    unique_values[i]->sample_given_data(curr_data);
  }
}
