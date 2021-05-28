#ifndef BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "conditional_algorithm.h"

//! Template class for the blocked Gibbs sampling algorithm.

//! This class implement the blocked Gibbs sampling procedure from Ishwaran and
//! James (2004). A basic example of conditional algorithm, it performs
//! allocation sampling based on the mixing weights and on the clusters'
//! likelihood given their own unique values. Updates of unique values for each
//! cluster are performed similarly to marginal algorithms, with a Bayesian
//! posterior update given all observations belonging to that cluster. Finally,
//! the weights update phase heavily depends on the `Mixing` object use -- in
//! fact, an appropriate method from that object is called.
//! For more information, please refer to the `BaseAlgorithm` and
//! `ConditionalAlgorithm` classes, from which this one inherits.

class BlockedGibbsAlgorithm : public ConditionalAlgorithm {
 public:
  BlockedGibbsAlgorithm() = default;
  ~BlockedGibbsAlgorithm() = default;

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::BlockedGibbs;
  }

 protected:
  void print_startup_message() const override;

  void sample_allocations() override;

  void sample_unique_values() override;

  void sample_weights() override;
};

#endif  // BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
