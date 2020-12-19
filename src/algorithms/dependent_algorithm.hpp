#ifndef BAYESMIX_ALGORITHMS_DEPENDENT_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_DEPENDENT_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "marginal_algorithm.hpp"

class DependentAlgorithhm : public MarginalAlgorithm {
 protected:
  Eigen::MatrixXd covariates = Eigen::MatrixXd(0,0);

  void add_datum_to_hierarchy(BaseHierarchy *hier, const int idx) override;

  // ALGORITHM FUNCTIONS
  void initialize() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~DependentAlgorithhm() = default;
  DependentAlgorithhm() = default;

  void set_covariates(const Eigen::MatrixXd &cov_) { covariates = cov_; }
};

#endif  // BAYESMIX_ALGORITHMS_DEPENDENT_ALGORITHM_HPP_
