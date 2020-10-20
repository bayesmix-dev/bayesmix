#ifndef CONDITIONALALGORITHM_HPP
#define CONDITIONALALGORITHM_HPP

#include "Algorithm.hpp"

class ConditionalAlgorithm : public Algorithm {
public:
  MarginalAlgorithm(const BaseMixing &mixing_,
                    const Eigen::MatrixXd &data_, const unsigned int init = 0)
      : Algorithm::Algorithm(mixing_, data_, init) {}
  virtual void eval_density(const Eigen::MatrixXd &grid,
                            BaseCollector *const collector) override;
};

#endif  // CONDITIONALALGORITHM_HPP
