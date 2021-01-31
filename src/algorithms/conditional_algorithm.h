#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>

#include "base_algorithm.hpp"
#include "src/collectors/base_collector.hpp"

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const coll) override;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_