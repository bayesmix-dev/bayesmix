#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"

/**
 * Template class for a conditional sampler deriving from `BaseAlgorithm`.
 *
 * This template class implements a generic Gibbs sampling conditional
 * algorithm as the child of the `BaseAlgorithm` class.
 * A mixture model sampled from a conditional algorithm can be expressed as
 *
 * \f[
 *    x_i \mid c_i, \theta_1, \dots, \theta_k &\sim f(x_i \mid \theta_{c_i}) \\
 *    \theta_1, \dots, \theta_k &\sim G_0 \\
 *    c_1, \dots, c_n \mid w_1, \dots, w_k &\sim \text{Cat}(w_1, \dots, w_k) \\
 *    w_1, \dots, w_k &\sim p(w_1, \dots, w_k)
 * \f]
 *
 * where \f$ f(x \mid \theta_j) \f$ is a density for each value of \f$ \theta_j
 * \f$, \f$ c_i \f$ take values in \f$ \{1, \dots, k\} \f$ and \f$ w_1, \dots,
 * w_k \f$ are nonnegative weights whose sum is a.s. 1, i.e. \f$ p(w_1, ...
 * w_k) \f$ is a probability distribution on the k-1 dimensional unit simplex).
 * In this library, each \f$ \theta_j \f$ is represented as an `Hierarchy`
 * object (which inherits from `AbstractHierarchy`), that also holds the
 * information related to the base measure \f$ G \f$ is (see
 * `AbstractHierarchy`). The weights \f$ (w_1, \dots, w_k) \f$ are represented
 * as a `Mixing` object, which inherits from `AbstractMixing`.
 *
 * The state of a conditional algorithm consists of the unique values, the
 * cluster allocations and the mixture weights. The former two are stored in
 * this class, while the weights are stored in the `Mixing` object.
 */

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ConditionalAlgorithm() = default;
  ~ConditionalAlgorithm() = default;

  bool is_conditional() const override { return true; }

  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) override;

 protected:
  //! Performs Gibbs sampling sub-step for all component weights
  virtual void sample_weights() = 0;

  void step() override {
    sample_allocations();
    sample_unique_values();
    update_hierarchy_hypers();
    sample_weights();
  }
};

#endif  // BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
