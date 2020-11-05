#ifndef CONDITIONALALGORITHM_HPP
#define CONDITIONALALGORITHM_HPP

#include "Algorithm.hpp"

class ConditionalAlgorithm : public Algorithm {
 protected:
  virtual void print_startup_message() const override = 0;

 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const collector) override;
};

#endif  // CONDITIONALALGORITHM_HPP
