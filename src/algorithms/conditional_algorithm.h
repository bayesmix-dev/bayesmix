#ifndef BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_CONDITIONAL_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "base_algorithm.h"
#include "src/collectors/base_collector.h"

//! Template class for a conditional sampler deriving from `BaseAlgorithm`.

//! This template class implements a generic Gibbs sampling conditional
//! algorithm as the child of the `BaseAlgorithm` class.
//! A mixture model sampled from a Marginal Algorithm can be expressed as
//!   x_i | c_i, phi_1, ..., phi_k ~ f(x_i|phi_(c_i))    (data likelihood);
//!               phi_1, ... phi_k ~ G                   (unique values);
//!   c_1, ... c_n | w_1, ..., w_k ~ Cat(w_1, ... w_k)   (cluster allocations);
//!                  w_1, ..., w_k ~ p(w_1, ..., w_k)    (mixture weights)
//! where f(x | phi_j) is a density for each value of phi_j, the c_i take
//! values in {1, ..., k} and w_1, ..., w_k are nonnegative weight that sum
//! to one almost surely (i.e. p(w_1, ... w_k) is a probability distribution
//! on the k-1 dimensional unit simplex).
//! In this library, each phi_j is represented as an `Hierarchy` object
//! (inheriting from `AbstractHierarchy`), that also knows what the base
//! measure `G` is (see `AbstractHierarchy`).
//! The weights (w_1, ..., w_k) are represented as a `Mixing` object
//! (inheriting from `AbstractMixing`).

//! The state of a conditional algorithm consists of the unique values, the
//! cluster allocations and the mixture weights. The first two are stored
//! in this class, while the weights are stored in the `Mixing` object.

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
