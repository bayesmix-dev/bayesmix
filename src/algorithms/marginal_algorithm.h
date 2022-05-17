#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/abstract_hierarchy.h"

/**
 * Template class for a marginal sampler deriving from `BaseAlgorithm`.
 *
 * This template class implements a generic Gibbs sampling marginal algorithm
 * as the child of the `BaseAlgorithm` class.
 * A mixture model sampled from a Marginal Algorithm can be expressed as
 *
 * \f[
 *      x_i \mid c_i, \theta_1, \dots, \theta_k &\sim f(x_i \mid \theta_{c_i})
 * \\
 *      \theta_1, \dots, \theta_k &\sim G_0 \\
 *      c_1, \dots, c_n &\sim EPPF(c_1, \dots, c_n)
 * \f]
 *
 * where \f$ f(x \mid \theta_j) \f$ is a density for each value of \f$ \theta_j
 * \f$ and \f$ c_i \f$ take values in \f$ {1, \dots, k} \f$. Depending on the
 * actual implementation, the algorithm might require the kernel/likelihood \f$
 * f(x \mid \theta) \f$ and \f$ G_0(\phi) \f$ to be conjugagte or not. In the
 * former case, a conjugate hierarchy must be specified. In this library, each
 * \f$ \theta_j \f$ is represented as an `Hierarchy` object (which inherits
 * from `AbstractHierarchy`), that also holds the information related to the
 * base measure \f$ G_0 \f$ is (see `AbstractHierarchy`). The \f$ EPPF \f$ is
 * instead represented as a `Mixing` object, which inherits from
 * `AbstractMixing`.
 *
 * The state of a marginal algorithm only consists of allocations and unique
 * values. In this class of algorithms, the local lpdf estimate for a single
 * iteration is a weighted average of likelihood values corresponding to each
 * component (i.e. cluster), with the weights being based on its cardinality,
 * and of the marginal component, which depends on the specific algorithm.
 */

class MarginalAlgorithm : public BaseAlgorithm {
 public:
  MarginalAlgorithm() = default;
  ~MarginalAlgorithm() = default;

  bool is_conditional() const override { return false; }

  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) override;

 protected:
  //! Computes marginal contribution of the given cluster to the lpdf estimate
  //! @param hier       Pointer to the `Hierarchy` object representing the
  //! cluster
  //! @param grid       Grid of row points on which the density is to be
  //! evaluated
  //! @param covariate  (Optional) covariate vectors associated to data
  //! @return           The marginal component of the estimate
  virtual Eigen::VectorXd lpdf_marginal_component(
      const std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const = 0;

  //! Deletes cluster from the algorithm state given its label
  void remove_singleton(const unsigned int idx);
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_H_
