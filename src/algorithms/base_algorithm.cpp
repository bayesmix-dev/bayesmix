#include "base_algorithm.hpp"

#include <Eigen/Dense>
#include <cassert>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../../proto/cpp/mixings.pb.h"

void BaseAlgorithm::initialize() {
  std::cout << "Initializing..." << std::endl;

  // Perform checks
  assert(data.rows() != 0 && "Error: empty data matrix");
  assert(unique_values.size() != 0 && "Error: hierarchy was not provided");
  assert(!(unique_values[0]->is_multivariate() == false && data.cols() > 1) &&
         "Error: multivariate data supplied to univariate hierarchy");

  if (init_num_clusters == 0) {
    init_num_clusters = data.rows();
  }

  // Initialize hierarchies
  unique_values[0]->initialize();
  for (size_t i = 0; i < init_num_clusters - 1; i++) {
    unique_values.push_back(unique_values[0]->clone());
  }

  // Initialize needed objects
  cardinalities.clear();
  cardinalities.reserve(data.rows());
  std::default_random_engine generator;
  // Build uniform probability on clusters, given their initial number
  std::uniform_int_distribution<int> distro(0, init_num_clusters - 1);
  // Allocate one datum per cluster first, and update cardinalities
  allocations.clear();
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

void BaseAlgorithm::update_hierarchy_hypers() {
  bayesmix::MarginalState::ClusterVal clust;
  std::vector<bayesmix::MarginalState::ClusterVal> states;
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
  *iter_out.mutable_cluster_cards() = {cardinalities.begin(),
                                       cardinalities.end()};
  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    bayesmix::MarginalState::ClusterVal clusval;
    unique_values[i]->write_state_to_proto(&clusval);
    iter_out.add_cluster_vals()->CopyFrom(clusval);
  }

  // Transcribe mixing state
  bayesmix::MixingState mixstate;
  mixing->write_state_to_proto(&mixstate);
  iter_out.add_mixing_states()->CopyFrom(mixstate);

  return iter_out;
}
