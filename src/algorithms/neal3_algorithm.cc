#include "neal3_algorithm.h"

#include <Eigen/Dense>
#include <memory>

#include "hierarchy_id.pb.h"
#include "mixing_id.pb.h"
#include "src/hierarchies/base_hierarchy.h"

Eigen::VectorXd Neal3Algorithm::get_cluster_lpdf(
    const unsigned int data_idx) const {
  unsigned int n_data = data.rows();
  unsigned int n_clust = unique_values.size();
  Eigen::VectorXd loglpdf(n_clust + 1);
  for (size_t j = 0; j < n_clust; j++) {
    // Probability of being assigned to an already existing cluster
    loglpdf(j) = unique_values[j]->conditional_pred_lpdf(
        data.row(data_idx), hier_covariates.row(data_idx));
  }
  // Probability of being assigned to a newly created cluster
  loglpdf(n_clust) = unique_values[0]->prior_pred_lpdf(
      data.row(data_idx), hier_covariates.row(data_idx));
  return loglpdf;
}

void Neal3Algorithm::print_startup_message() const {
  std::string msg = "Running Neal3 algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) +
                    " mixing...";
  std::cout << msg << std::endl;
}
