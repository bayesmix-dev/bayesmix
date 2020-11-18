#include "base_algorithm.hpp"

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"

//! \param iter Number of the current iteration
//! \return     Protobuf-object version of the current state
bayesmix::MarginalState BaseAlgorithm::get_state_as_proto(unsigned int iter) {
  bayesmix::MarginalState iter_out;
  // Transcribe iteration number and allocations vector
  iter_out.set_iteration_num(iter);
  *iter_out.mutable_cluster_allocs() = {allocations.begin(),
                                        allocations.end()};

  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    bayesmix::MarginalState::ClusterVal clusval;
    unique_values[i]->write_state_to_proto(&clusval);
    iter_out.add_cluster_vals()->CopyFrom(clusval);
  }

  // Transcribe total mass from mixing object
  // ...

  return iter_out;
}
