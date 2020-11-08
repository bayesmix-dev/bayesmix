#ifndef MARGINALALGORITHM_HPP
#define MARGINALALGORITHM_HPP

#include "algorithm_base.hpp"

class AlgorithmMarginal : public AlgorithmBase {
 protected:
  virtual void print_startup_message() const override = 0;

 public:
  ~AlgorithmMarginal() = default;
  AlgorithmMarginal() = default;
  virtual Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                                    CollectorBase *const collector) override;
};

#endif  // MARGINALALGORITHM_HPP
