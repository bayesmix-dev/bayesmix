#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_DEP_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_DEP_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "conditional_algorithm.hpp"

class ConditionalDepAlgorithm : public ConditionalAlgorithm {
 protected:
  Eigen::MatrixXd covariates = Eigen::MatrixXd(0, 0);

  // ALGORITHM FUNCTIONS
  void initialize() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~ConditionalDepAlgorithm() = default;
  ConditionalDepAlgorithm() = default;

  void set_covariates(const Eigen::MatrixXd &cov_) { covariates = cov_; }
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_DEP_ALGORITHM_HPP_
