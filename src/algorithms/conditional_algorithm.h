#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"

//! Template class for a conditional sampler deriving from `BaseAlgorithm`.

//! This template class implements a generic Gibbs sampling conditional
//! algorithm as the child of the `BaseAlgorithm` class.
//!
//! A conditional algorithm is one in which the component weights are not
//! marginalized out, unlike in the marginal case. Weights therefore are part
//! of the state of the algorithm alongside allocations and unique values.
//! Although, unlike the latter two, weights are not stored in this class, but
//! into the pointed `Mixing` object. In this class of algorithms, the local
//! lpdf estimate for a single iteration is a weighted average of likelihood
//! values corresponding to each component (i.e. cluster), with respect to the
//! state weights described earlier.
//! For more information on Gibbs samplers implemented in the library, please
//! refer to the `BaseAlgorithm` base class.

class ConditionalAlgorithm : public BaseAlgorithm {
 public:
  ConditionalAlgorithm() = default;
  ~ConditionalAlgorithm() = default;

  bool is_conditional() const override { return true; }

 protected:
  Eigen::VectorXd lpdf_from_state(
      const Eigen::MatrixXd &grid, const Eigen::RowVectorXd &hier_covariate,
      const Eigen::RowVectorXd &mix_covariate) override;

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
