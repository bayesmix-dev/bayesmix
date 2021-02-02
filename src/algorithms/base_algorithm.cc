#include "base_algorithm.h"

#include <Eigen/Dense>
#include <memory>

#include "marginal_state.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/hierarchies/dependent_hierarchy.h"
#include "src/mixings/dependent_mixing.h"

void BaseAlgorithm::add_datum_to_hierarchy(
    const unsigned int datum_idx, std::shared_ptr<BaseHierarchy> hier) {
  if (hier->is_dependent()) {
    auto hiercast = std::dynamic_pointer_cast<DependentHierarchy>(hier);
    hiercast->add_datum(datum_idx, data.row(datum_idx),
                        hier_covariates.row(datum_idx));
  } else {
    hier->add_datum(datum_idx, data.row(datum_idx));
  }
}

void BaseAlgorithm::remove_datum_from_hierarchy(
    const unsigned int datum_idx, std::shared_ptr<BaseHierarchy> hier) {
  if (hier->is_dependent()) {
    auto hiercast = std::dynamic_pointer_cast<DependentHierarchy>(hier);
    hiercast->remove_datum(datum_idx, data.row(datum_idx),
                           hier_covariates.row(datum_idx));
  } else {
    hier->remove_datum(datum_idx, data.row(datum_idx));
  }
}

void BaseAlgorithm::initialize() {
  std::cout << "Initializing... " << std::flush;
  // Data check
  if (data.rows() == 0) {
    throw std::invalid_argument("Data was not provided to algorithm");
  }
  // Hierarchy checks
  if (unique_values.size() == 0) {
    throw std::invalid_argument("Hierarchy was not provided to algorithm");
  }
  if (unique_values[0]->is_multivariate() == false && data.cols() > 1) {
    throw std::invalid_argument(
        "Multivariate data supplied to univariate hierarchy");
  }
  if (hier_covariates.rows() != 0) {
    if (unique_values[0]->is_dependent() == false) {
      throw std::invalid_argument(
          "Covariates supplied to non-dependent hierarchy");
    }
    if (data.rows() != hier_covariates.rows()) {
      throw std::invalid_argument(
          "Sizes of data and hierarchy covariates do not match");
    }
  }
  else {
    // Create empty covariates vector
    hier_covariates = Eigen::MatrixXd::Zero(data.rows(), 0);
  }
  // Mixing checks
  if (mixing == nullptr) {
    throw std::invalid_argument("Mixing was not provided to algorithm");
  }
  if (mix_covariates.rows() != 0) {
    if (mixing->is_dependent() == false) {
      throw std::invalid_argument(
          "Covariates supplied to non-dependent mixing");
    }
    if (data.rows() != mix_covariates.rows()) {
      throw std::invalid_argument(
          "Sizes of data and mixing covariates do not match");
    }
  }
  else {
    // Create empty covariates vector
    mix_covariates = Eigen::MatrixXd::Zero(data.rows(), 0);
  }
  // Interpet default number of clusters
  if (init_num_clusters == 0) {
    init_num_clusters = data.rows();
  }
  // Initialize hierarchies
  unique_values[0]->initialize();
  for (size_t i = 0; i < init_num_clusters - 1; i++) {
    unique_values.push_back(unique_values[0]->clone());
    unique_values[i]->draw();
  }
  // Initialize mixing
  mixing->initialize();
  // Build uniform probability on clusters, given their initial number
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distro(0, init_num_clusters - 1);
  // Allocate one datum per cluster first, and update cardinalities
  allocations.clear();
  for (size_t i = 0; i < init_num_clusters; i++) {
    allocations.push_back(i);
    add_datum_to_hierarchy(i, unique_values[i]);
  }
  // Randomly allocate all remaining data, and update cardinalities
  for (size_t i = init_num_clusters; i < data.rows(); i++) {
    unsigned int clust = distro(generator);
    allocations.push_back(clust);
    add_datum_to_hierarchy(i, unique_values[clust]);
  }
  std::cout << "Done" << std::endl;
}

void BaseAlgorithm::update_hierarchy_hypers() {
  bayesmix::MarginalState::ClusterState clust;
  std::vector<bayesmix::MarginalState::ClusterState> states;
  for (auto &un : unique_values) {
    un->write_state_to_proto(&clust);
    states.push_back(clust);
  }
  unique_values[0]->update_hypers(states);
}

//! \param iter Number of the current iteration
//! \return     Protobuf-object version of the current state
bayesmix::MarginalState BaseAlgorithm::get_state_as_proto(unsigned int iter) {
  bayesmix::MarginalState iter_out;
  // Transcribe iteration number, allocations, and cardinalities
  iter_out.set_iteration_num(iter);
  *iter_out.mutable_cluster_allocs() = {allocations.begin(),
                                        allocations.end()};
  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    bayesmix::MarginalState::ClusterState clusval;
    unique_values[i]->write_state_to_proto(&clusval);
    iter_out.add_cluster_states()->CopyFrom(clusval);
  }

  // Transcribe mixing state
  bayesmix::MixingState mixstate;
  mixing->write_state_to_proto(&mixstate);
  iter_out.mutable_mixing_state()->CopyFrom(mixstate);

  return iter_out;
}
