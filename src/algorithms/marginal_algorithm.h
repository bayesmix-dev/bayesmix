#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! TODO class description

class MarginalAlgorithm : public BaseAlgorithm {
 public:
  MarginalAlgorithm() = default;
  ~MarginalAlgorithm() = default;

  bool is_conditional() const override { return false; }

 protected:
  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) override;

  //! Computes marginal contribution of the given cluster to the lpdf estimate
  //! @param hier   Pointer to the `Hierarchy` object representing the cluster
  //! @param grid   Grid of row points on which the density is to be evaluated
  //! @return       The marginal component of the estimate
  virtual Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const = 0;

  //! Deletes cluster from the algorithm state given its label
  void remove_singleton(const unsigned int idx);
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
