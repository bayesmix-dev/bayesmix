#include "blocked_gibbs_algorithm.h"

#include "hierarchy_id.pb.h"
#include "mixing_id.pb.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/mixings/base_mixing.h"

void BlockedGibbsAlgorithm::print_startup_message() const {
  std::string msg = "Running BlockedGibbs algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}

void BlockedGibbsAlgorithm::sample_allocations() {
  return;  // TODO
}

void BlockedGibbsAlgorithm::sample_unique_values() {
  return;  // TODO
}
