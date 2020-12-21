#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "base_algorithm.hpp"

class MarginalAlgorithm : public BaseAlgorithm {
 public:
  ~MarginalAlgorithm() = default;
  MarginalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(
      const Eigen::MatrixXd &grid,
      BaseCollector<bayesmix::MarginalState> *const coll) override;
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_
