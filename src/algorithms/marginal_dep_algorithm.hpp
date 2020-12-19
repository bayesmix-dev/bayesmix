#ifndef BAYESMIX_ALGORITHMS_MARGINAL_DEP_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_MARGINAL_DEP_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "marginal_algorithm.hpp"

class MarginalDepAlgorithhm : public MarginalAlgorithm {
 protected:
  Eigen::MatrixXd covariates = Eigen::MatrixXd(0, 0);

  // ALGORITHM FUNCTIONS
  void initialize() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~MarginalDepAlgorithhm() = default;
  MarginalDepAlgorithhm() = default;

  void set_covariates(const Eigen::MatrixXd &cov_) { covariates = cov_; }
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_DEP_ALGORITHM_HPP_
