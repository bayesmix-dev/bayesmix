#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../collectors/base_collector.hpp"
#include "base_algorithm.hpp"

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const coll) override;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
