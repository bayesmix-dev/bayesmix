#ifndef MARGINALALGORITHM_HPP
#define MARGINALALGORITHM_HPP

#include "Algorithm.hpp"

class MarginalAlgorithm : public Algorithm {
 protected:
  virtual void print_startup_message() const override = 0;

 public:
  ~MarginalAlgorithm() = default;
  MarginalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const collector) override;
};

#endif  // MARGINALALGORITHM_HPP
