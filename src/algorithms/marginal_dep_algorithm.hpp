#ifndef BAYESMIX_ALGORITHMS_MARGINAL_DEP_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_MARGINAL_DEP_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "marginal_algorithm.hpp"

class MarginalDepAlgorithm : public MarginalAlgorithm {
 protected:
  Eigen::MatrixXd covariates;

  // ALGORITHM FUNCTIONS
  void initialize() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~MarginalDepAlgorithm() = default;
  MarginalDepAlgorithm() = default;

  Eigen::MatrixXd eval_lpdf(
      const Eigen::MatrixXd &grid,
      BaseCollector<bayesmix::MarginalState> *const coll) override;

  void set_covariates(const Eigen::MatrixXd &cov_) { covariates = cov_; }
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_DEP_ALGORITHM_HPP_
