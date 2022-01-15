#include "split_and_merge_algorithm.h"

void SplitAndMergeAlgorithm::print_startup_message() const {
  std::string msg = "Running Split and Merge algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}