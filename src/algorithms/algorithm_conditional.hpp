#ifndef CONDITIONALALGORITHM_HPP
#define CONDITIONALALGORITHM_HPP

#include "algorithm_base.hpp"

class AlgorithmConditional : public AlgorithmBase {
 protected:
  virtual void print_startup_message() const override = 0;

 public:
  ~AlgorithmConditional() = default;
  AlgorithmConditional() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const collector) override;
};

#endif  // CONDITIONALALGORITHM_HPP
