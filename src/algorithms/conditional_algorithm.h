#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/conditional_mixing.h"

class ConditionalAlgorithm : public BaseAlgorithm {
 protected:
  //! Weights of clusters
  Eigen::VectorXd weights;
  //! Points at the same object as BaseAlgorithm::mixing
  ConditionalMixing cond_mixing;

 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  //!
  void initialize() override;
  //!
  void step() override {
    BaseAlgorithm::step();
    cond_mixing->update_state(unique_values, data.size());
  }
  //!
  virtual Eigen::MatrixXd eval_lpdf(
      BaseCollector *const collector, const Eigen::MatrixXd &grid,
      const Eigen::MatrixXd &hier_covariates = Eigen::MatrixXd(0, 0),
      const Eigen::MatrixXd &mix_covariates =
          Eigen::MatrixXd(0, 0)) const override;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
