#include "neal2_dep_algorithm.hpp"

#include <Eigen/Dense>
#include <memory>

#include "../hierarchies/base_hierarchy.hpp"
#include "neal2_algorithm.hpp"

void Neal2DepAlgorithm::print_startup_message() const {
  std::string msg = "Running Neal2 dependent algorithm with " +
                    unique_values[0]->get_id() + " hierarchies, " +
                    mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

void Neal2DepAlgorithm::lpdf_marginal_component(
      std::shared_ptr<BaseHierarchy> temp_hier,
      const std::vector<int> &idxs) {
  Neal2::lpdf_marginal_component(temp_hier, idxs);
}

void Neal2DepAlgorithm::sample_allocations() {
  Neal2::sample_allocations();
}

void Neal2DepAlgorithm::sample_unique_values() {
  Neal2::sample_unique_values();
}
