#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Template class for a marginal sampler deriving from `BaseAlgorithm`.

//! This template class implements a generic Gibbs sampling marginal algorithm
//! as the child of the `BaseAlgorithm` class.
//!
//! A marginal algorithm is one in which the component weights have been
//! marginalized out. Therefore its state only consists of allocations and
//! unique values. In this class of algorithms, the local lpdf estimate for
//! a single iteration is a weighted average of likelihood values corresponding
//! to each component (i.e. cluster), with the weights being based on its
//! cardinality, and of the marginal component, which depends on the specific.
//! algorithm.
//! For more information on Gibbs samplers implemented in the library, please
//! refer to the `BaseAlgorithm` base class.

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
