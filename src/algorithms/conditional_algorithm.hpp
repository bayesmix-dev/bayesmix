#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../collectors/base_collector.hpp"
#include "base_algorithm.hpp"

class ConditionalAlgorithm : public BaseAlgorithm {
 protected:
  virtual void print_startup_message() const override = 0;

 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const collector) override;
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_HPP_
