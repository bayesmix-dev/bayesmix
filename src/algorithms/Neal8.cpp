#include "Neal8.hpp"

//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal8::density_marginal_component(
    std::shared_ptr<HierarchyBase> temp_hier) {
  Eigen::VectorXd dens_addendum(density.first.rows());
  // Loop over unique values for a "sample mean" of the marginal
  for (size_t i = 0; i < n_aux; i++) {
    // Generate unique values from their prior centering distribution
    temp_hier->draw();
    dens_addendum += temp_hier->like(density.first) / n_aux;
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
      aux_unique_values[0]->set_state(
          unique_values[allocations[i]]->get_state(), false);
      singleton = 1;
    }

    // Remove datum from cluster
    cardinalities[allocations[i]] -= 1;

    // Draw the unique values in the auxiliary blocks from their prior
    for (size_t j = singleton; j < n_aux; j++) {
      aux_unique_values[j]->draw();
    }

    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas(n_clust + n_aux);
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      logprobas(j) =
          log(mixing->mass_existing_cluster(cardinalities[j], n - 1)) +
          unique_values[j]->lpdf(datum)(0);
      // Note: if datum is a singleton, then, when j = allocations[i],
      // one has card[j] = 0: cluster j will never be chosen
    }
    // Loop over auxiliary blocks
    for (size_t j = 0; j < n_aux; j++) {
      // Probability of being assigned to a newly generated cluster
      logprobas(n_clust + j) = log(mixing->mass_new_cluster(n_clust, n - 1)) +
                               aux_unique_values[j]->lpdf(datum)(0) -
                               log(n_aux);
    }
    // Draw a NEW value for datum allocation
    auto rng = bayesmix::Rng::Instance().get();
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);

    // Assign datum to its new cluster and update cardinalities:
    // 4 cases are handled separately
    if (singleton == 1) {
      if (c_new >= n_clust) {
        // Case 1: datum moves from a singleton to a new cluster
        // Take unique values from an auxiliary block
        unique_values[allocations[i]]->set_state(
            aux_unique_values[c_new - n_clust]->get_state(), false);
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
