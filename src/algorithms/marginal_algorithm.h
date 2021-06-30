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
//! A mixture model sampled from a Marginal Algorithm can be expressed as
//!   x_i | c_i, phi_1, ..., phi_k ~ f(x_i|phi_(c_i))    (data likelihood);
//!               phi_1, ... phi_k ~ G                   (unique values);
//!                   c_1, ... c_n ~ EPPF(c_1, ... c_n)  (cluster allocations);
//! where f(x | phi_j) is a density for each value of phi_j and the c_i take
//! values in {1, ..., k}.
//! Depending on the actual implementation, the algorithm might require
//! the kernel/likelihood f(x | phi) and G(phi) to be conjugagte or not.
//! In the former case, a `ConjugateHierarchy` must be specified.
//! In this library, each phi_j is represented as an `Hierarchy` object (which
//! inherits from `AbstractHierarchy`), that also holds the information related
//! to the base measure `G` is (see `AbstractHierarchy`). The EPPF is instead
//! represented as a `Mixing` object, which inherits from `AbstractMixing`.
//!
//! The state of a marginal algorithm only consists of allocations and unique
//! values. In this class of algorithms, the local lpdf estimate for a single
//! iteration is a weighted average of likelihood values corresponding to each
//! component (i.e. cluster), with the weights being based on its cardinality,
//! and of the marginal component, which depends on the specific algorithm.

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
