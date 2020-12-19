#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "base_algorithm.hpp"

class MarginalAlgorithm : public BaseAlgorithm {
 protected:
  //! Computes marginal contribution of a given iteration & cluster
  virtual Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<BaseHierarchy> temp_hier,
      const Eigen::MatrixXd &grid) = 0;

 public:
  ~MarginalAlgorithm() = default;
  MarginalAlgorithm() = default;
  virtual Eigen::MatrixXd eval_lpdf(
      const Eigen::MatrixXd &grid,
      BaseCollector<bayesmix::MarginalState> *const coll) override;
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_
