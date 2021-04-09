#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/abstract_hierarchy.h"

class MarginalAlgorithm : public BaseAlgorithm {
 protected:
  //! Computes marginal contribution of a given iteration & cluster
  virtual Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const = 0;
  //!
  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) override;
  //!
  void remove_singleton(const unsigned int idx);

 public:
  ~MarginalAlgorithm() = default;
  MarginalAlgorithm() = default;
  //!
  bool is_conditional() const override { return false; }
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
