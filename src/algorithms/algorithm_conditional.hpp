#ifndef BAYESMIX_ALGORITHMS_ALGORITHM_CONDITIONAL_HPP_
#define BAYESMIX_ALGORITHMS_ALGORITHM_CONDITIONAL_HPP_

#include "algorithm_base.hpp"

class AlgorithmConditional : public AlgorithmBase {
 protected:
  virtual void print_startup_message() const override = 0;

 public:
  ~AlgorithmConditional() = default;
  AlgorithmConditional() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    CollectorBase *const collector) override;
};

#endif  // BAYESMIX_ALGORITHMS_ALGORITHM_CONDITIONAL_HPP_
