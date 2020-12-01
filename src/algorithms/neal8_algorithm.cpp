#include "neal8_algorithm.hpp"

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../hierarchies/base_hierarchy.hpp"
#include "../mixings/base_mixing.hpp"
#include "../utils/distributions.hpp"
#include "neal2_algorithm.hpp"

//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal8Algorithm::lpdf_marginal_component(
    std::shared_ptr<BaseHierarchy> temp_hier, const Eigen::MatrixXd &grid) {
  unsigned int n_grid = grid.rows();
  Eigen::VectorXd lpdf_(n_grid);
  Eigen::MatrixXd lpdf_temp(n_grid, n_aux);
  // Loop over unique values for a "sample mean" of the marginal
  for (size_t i = 0; i < n_aux; i++) {
    // Generate unique values from their prior centering distribution
    temp_hier->draw();
    lpdf_temp.col(i) = temp_hier->like_lpdf_grid(grid);
  }
  for (size_t i = 0; i < n_grid; i++) {
    lpdf_(i) = stan::math::log_sum_exp(lpdf_temp.row(i));
  }
  return lpdf_.array() - log(n_aux);
}

void Neal8Algorithm::print_startup_message() const {
  std::string msg = "Running Neal8 algorithm (m=" + std::to_string(n_aux) +
                    " aux. blocks) with " + unique_values[0]->get_id() +
                    " hierarchies, " + mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

void Neal8Algorithm::initialize() {
  Neal2Algorithm::initialize();
  // Create correct amount of auxiliary blocks
  aux_unique_values.clear();
  for (size_t i = 0; i < n_aux; i++) {
    aux_unique_values.push_back(unique_values[0]->clone());
  }
}

void Neal8Algorithm::sample_allocations() {
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
    int singleton = int(unique_values[allocations[i]]->get_card() == 1);
    if (singleton) {
      // Save unique value in the first auxiliary block
      bayesmix::MarginalState::ClusterState curr_val;
      unique_values[allocations[i]]->write_state_to_proto(&curr_val);
      aux_unique_values[0]->set_state_from_proto(curr_val);
    }

    // Remove datum from cluster
    unique_values[allocations[i]]->remove_datum(i, datum);

    // Draw the unique values in the auxiliary blocks from their prior
    for (size_t j = singleton; j < n_aux; j++) {
      aux_unique_values[j]->draw();
    }

    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas(n_clust + n_aux);
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      logprobas(j) = mixing->mass_existing_cluster(unique_values[j],
                                                   n_data - 1, true, true) +
                     unique_values[j]->like_lpdf(datum);
      // Note: if datum is a singleton, then, when j = allocations[i],
      // one has card[j] = 0: cluster j will never be chosen
    }
    // Loop over auxiliary blocks
    for (size_t j = 0; j < n_aux; j++) {
      // Probability of being assigned to a newly generated cluster
      logprobas(n_clust + j) =
          mixing->mass_new_cluster(n_clust, n_data - 1, true, true) +
          aux_unique_values[j]->like_lpdf(datum) - log(n_aux);
    }
    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
    unsigned int c_old = allocations[i];

    if (c_new >= n_clust) {
      // datum moves to a new cluster
      // Copy one of the auxiliary block as the new cluster
      std::shared_ptr<BaseHierarchy> hier_new =
          aux_unique_values[c_new - n_clust]->clone();
      unique_values.push_back(hier_new);
      allocations[i] = n_clust;
      unique_values[allocations[i]]->add_datum(i, datum);
    } else {
      allocations[i] = c_new;
      unique_values[allocations[i]]->add_datum(i, datum);
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
