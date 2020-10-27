#ifndef MARGINALALGORITHM_HPP
#define MARGINALALGORITHM_HPP

#include "Algorithm.hpp"

class MarginalAlgorithm : public Algorithm {
 public:
  ~MarginalAlgorithm() = default;
  MarginalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    BaseCollector *const collector) override;
};

#endif  // MARGINALALGORITHM_HPP
