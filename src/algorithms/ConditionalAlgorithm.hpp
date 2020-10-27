#ifndef CONDITIONALALGORITHM_HPP
#define CONDITIONALALGORITHM_HPP

#include "Algorithm.hpp"

class ConditionalAlgorithm : public Algorithm {
 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  virtual void eval_lpdf(const Eigen::MatrixXd &grid,
                            BaseCollector *const collector) override;
};

#endif  // CONDITIONALALGORITHM_HPP
