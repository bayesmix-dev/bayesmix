#ifndef MARGINALALGORITHM_HPP
#define MARGINALALGORITHM_HPP

#include "Algorithm.hpp"

class MarginalAlgorithm : public Algorithm {
 public:
  MarginalAlgorithm(const BaseMixing &mixing_,
                    const Eigen::MatrixXd &data_, const unsigned int init = 0)
      : Algorithm::Algorithm(mixing_, data_, init) {}
  virtual void eval_density(const Eigen::MatrixXd &grid,
                            BaseCollector *const collector) override;
};

#endif  // MARGINALALGORITHM_HPP
