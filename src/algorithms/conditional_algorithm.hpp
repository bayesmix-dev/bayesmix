#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "base_algorithm.hpp"

class ConditionalAlgorithm : public BaseAlgorithm {
 protected:
  //! Mixing weights
  Eigen::VectorXd weights;

  // ALGORITHM FUNCTIONS
  void initialize() override;
  virtual void sample_weights() = 0;

  //! Single step of algorithm
  void step() override {
    sample_allocations();
    sample_unique_values();
    sample_weights();
    update_hierarchy_hypers();
    mixing->update_state(unique_values, data.size());
  }

 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
