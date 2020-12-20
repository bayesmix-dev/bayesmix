#include "conditional_algorithm.hpp"

#include <Eigen/Dense>

#include "base_algorithm.hpp"

void ConditionalAlgorithm::initialize() {
  BaseAlgorithm::initialize();

  // Initialize weights
  weights = Eigen::VectorXd::Ones(init_num_clusters) / init_num_clusters;
}
