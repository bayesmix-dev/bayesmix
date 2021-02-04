#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    const Eigen::MatrixXd &hier_covariates,
                                    const Eigen::MatrixXd &mix_covariates,
                                    BaseCollector *const coll) const override;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
