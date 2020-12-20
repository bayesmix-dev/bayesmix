#include "probit_sb_algorithm.hpp"

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"

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

void ProbitSBAlgorithm::sample_weights() {
  return;  // TODO
}

Eigen::MatrixXd ProbitSBAlgorithm::eval_lpdf(
    const Eigen::MatrixXd &grid,
    BaseCollector<bayesmix::MarginalState> *const coll) {
  return Eigen::MatrixXd(0, 0);  // TODO
}
