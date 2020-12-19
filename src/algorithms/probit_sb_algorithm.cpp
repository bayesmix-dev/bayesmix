#include "probit_sb_algorithm.hpp"

void ProbitSBAlgorithm::print_startup_message() const {
  std::string msg = "Running ProbitStickBreak algorithm with " +
                    unique_values[0]->get_id() + " hierarchies, " +
                    mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

void ProbitSBAlgorithm::sample_allocations() {
  return;  // TODO
}

void ProbitSBAlgorithm::sample_unique_values() {
  return;  // TODO
}
