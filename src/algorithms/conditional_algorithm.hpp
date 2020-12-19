#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "base_algorithm.hpp"

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
