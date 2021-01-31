#include "base_algorithm.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <memory>

#include "marginal_state.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/base_hierarchy.hpp"
#include "src/hierarchies/dependent_hierarchy.hpp"
#include "src/mixings/dependent_mixing.hpp"

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

  // Perform checks
  assert(data.rows() != 0 && "Error: empty data matrix");
  assert(unique_values.size() != 0 && "Error: hierarchy was not provided");
  assert(!(unique_values[0]->is_multivariate() == false && data.cols() > 1) &&
         "Error: multivariate data supplied to univariate hierarchy");
  assert(mixing != nullptr && "Error: mixing was not provided");
  if (hier_covariates.rows() != 0) {
    assert(unique_values[0]->is_dependent() &&
           "Error: covariates supplied to non-dependent hierarchy");
    assert(data.rows() == hier_covariates.rows() &&
           "Error: data size and covariates size do not match");
  }
  if (mix_covariates.rows() != 0) {
    assert(mixing->is_dependent() &&
           "Error: covariates supplied to non-dependent mixing");
    assert(data.rows() == mix_covariates.rows() &&
           "Error: data size and covariates size do not match");
  }

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

  // Initialize needed objects
  std::default_random_engine generator;
  // Build uniform probability on clusters, given their initial number
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