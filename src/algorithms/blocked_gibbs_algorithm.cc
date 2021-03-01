#include "blocked_gibbs_algorithm.h"

#include <stan/math/prim/fun.hpp>

#include "hierarchy_id.pb.h"
#include "mixing_id.pb.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/mixings/base_mixing.h"
#include "src/utils/distributions.h"
#include "src/utils/rng.h"

void BlockedGibbsAlgorithm::print_startup_message() const {
  std::string msg = "Running BlockedGibbs algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(cond_mixing->get_id()) +
                    " mixing...";
  std::cout << msg << std::endl;
}

void BlockedGibbsAlgorithm::sample_allocations() {
  auto &rng = bayesmix::Rng::Instance().get();
  unsigned int num_components = cond_mixing->get_num_components();
  for (int i = 0; i < data.rows(); i++) {
    // Compute weights
    Eigen::VectorXd logprobas =
        cond_mixing->get_weights(true, false, mix_covariates.row(i));
    for (int j = 0; j < num_components; j++) {
      logprobas(j) +=
          unique_values[j]->like_lpdf(data.row(j), hier_covariates.row(j));
    }
    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
    unsigned int c_old = allocations[i];
    if (c_new != c_old) {
      allocations[i] = c_new;
      // Remove datum from old cluster, add to new
      unique_values[c_old]->remove_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
      unique_values[c_new]->add_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
    }
  }
}

void BlockedGibbsAlgorithm::sample_unique_values() {
  for (auto &un : unique_values) {
    un->sample_full_cond(!update_hierarchy_params());
  }
}

void BlockedGibbsAlgorithm::sample_weights() {
  cond_mixing->update_state(unique_values, allocations);
}
