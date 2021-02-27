#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/conditional_mixing.h"

class ConditionalAlgorithm : public BaseAlgorithm {
 protected:
  //! Points at the same object as BaseAlgorithm::mixing
  std::shared_ptr<ConditionalMixing> cond_mixing;
  //!
  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::MatrixXd &hier_covariates,
      const Eigen::MatrixXd &mix_covariates) override;
  //!
  void initialize() override;
  //!
  virtual void sample_weights() = 0;
  //!
  void step() override {
    sample_allocations();
    sample_unique_values();
    update_hierarchy_hypers();
    sample_weights();
  }

 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
