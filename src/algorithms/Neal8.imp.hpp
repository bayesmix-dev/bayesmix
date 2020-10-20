#ifndef NEAL8_IMP_HPP
#define NEAL8_IMP_HPP

#include "Neal8.hpp"

//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal8::density_marginal_component(
    HierarchyBase &temp_hier) {
  Eigen::VectorXd dens_addendum(density.first.rows());
  // Loop over unique values for a "sample mean" of the marginal
  for (size_t h = 0; h < n_aux; h++) {
    // Generate unique values from their prior centering distribution
    temp_hier.draw();
    dens_addendum += temp_hier.like(density.first) / n_aux;
  }
  return dens_addendum;
}

void Neal8::print_startup_message() const {
  std::cout << "Running Neal8 algorithm (with m=" << n_aux
            << " auxiliary blocks)..." << std::endl;
}

void Neal8::sample_allocations() {
  // Initialize relevant values
  unsigned int n = data.rows();

  // Loop over data points
  for (size_t i = 0; i < n; i++) {
    // Current i-th datum as row vector
    Eigen::Matrix<double, 1, Eigen::Dynamic> datum = data.row(i);
    // Initialize current number of clusters
    unsigned int n_clust = unique_values.size();
    // Initialize pseudo-flag
    int singleton = 0;
    if (cardinalities[allocations[i]] == 1) {
      // Save unique value in the first auxiliary block
      aux_unique_values[0].set_state(unique_values[allocations[i]].get_state(),
                                     false);
      singleton = 1;
    }

    // Remove datum from cluster
    cardinalities[allocations[i]] -= 1;

    // Draw the unique values in the auxiliary blocks from their prior
    for (size_t j = singleton; j < n_aux; j++) {
      aux_unique_values[j].draw();
    }

    // Compute probabilities of clusters
    Eigen::VectorXd probas(n_clust + n_aux);
    double tot = 0.0;
    // Loop over clusters
    for (size_t k = 0; k < n_clust; k++) {
      // Probability of being assigned to an already existing cluster
      probas(k) = mixing.mass_existing_cluster(cardinalities[k], n - 1) *
                  unique_values[k].like(datum)(0);
      tot += probas(k);
      // Note: if datum is a singleton, then, when k = allocations[i],
      // one has card[k] = 0: cluster k will never be chosen
    }
    // Loop over auxiliary blocks
    for (size_t k = 0; k < n_aux; k++) {
      // Probability of being assigned to a newly generated cluster
      probas(n_clust + k) = mixing.mass_new_cluster(n_clust, n - 1) *
                            aux_unique_values[k].like(datum)(0) / n_aux;
      tot += probas(n_clust + k);
    }
    // Normalize
    probas = probas / tot;

    // Draw a NEW value for datum allocation
    unsigned int c_new = stan::math::categorical_rng(probas, rng) - 1;

    // Assign datum to its new cluster and update cardinalities:
    // 4 cases are handled separately
    if (singleton == 1) {
      if (c_new >= n_clust) {
        // Case 1: datum moves from a singleton to a new cluster
        // Take unique values from an auxiliary block
        unique_values[allocations[i]].set_state(
            aux_unique_values[c_new - n_clust].get_state(), false);
        cardinalities[allocations[i]] += 1;
      } else {  // Case 2: datum moves from a singleton to an old cluster
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
      }  // end of else
    }    // end of if(singleton == 1)

    else {  // if singleton == 0
      if (c_new >= n_clust) {
        // Case 3: datum moves from a non-singleton to a new cluster
        // Copy one of the auxiliary block as the new cluster
        unique_values.push_back(aux_unique_values[c_new - n_clust]);
        cardinalities.push_back(1);
        allocations[i] = n_clust;
      } else {  // Case 4: datum moves from a non-singleton to an old cluster
        allocations[i] = c_new;
        cardinalities[c_new] += 1;
      }
    }
  }
}

#endif  // NEAL8_IMP_HPP
