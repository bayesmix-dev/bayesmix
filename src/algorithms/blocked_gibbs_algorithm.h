#ifndef BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "conditional_algorithm.h"

//! Template class for the blocked Gibbs sampling algorithm.

//! This class implement the blocked Gibbs sampling procedure from [1].
//!
//! [1] Ishwaran, H., & James, L. F. (2001). Gibbs sampling methods for
//! stick-breaking priors. Journal of the American Statistical
//! Association, 96(453), 161-173.

class BlockedGibbsAlgorithm : public ConditionalAlgorithm {
 public:
  BlockedGibbsAlgorithm() = default;
  ~BlockedGibbsAlgorithm() = default;

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::BlockedGibbs;
  }

  std::shared_ptr<BaseAlgorithm> clone() override {
    auto out = std::make_shared<BlockedGibbsAlgorithm>(*this);
    out->set_mixing(mixing->clone());
    out->set_hierarchy(unique_values[0]->clone());
    return out;
  }

 protected:
  void print_startup_message() const override;

  void sample_allocations() override;

  void sample_unique_values() override;

  void sample_weights() override;
};

#endif  // BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
