#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"

// TODO class description. Mention weights as part of state! (but they are not
// stored here)

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ConditionalAlgorithm() = default;
  ~ConditionalAlgorithm() = default;

  bool is_conditional() const override { return true; }

 protected:
  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) override;

  //! Performs Gibbs sampling sub-step for all component weights
  virtual void sample_weights() = 0;

  void step() override {
    sample_allocations();
    sample_unique_values();
    update_hierarchy_hypers();
    sample_weights();
  }
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
