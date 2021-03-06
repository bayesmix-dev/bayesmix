#ifndef BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "conditional_algorithm.h"

class BlockedGibbsAlgorithm : public ConditionalAlgorithm {
 protected:
  //!
  void print_startup_message() const override;
  //!
  void sample_allocations() override;
  //!
  void sample_unique_values() override;
  //!
  void sample_weights() override;

 public:
  ~BlockedGibbsAlgorithm() = default;
  BlockedGibbsAlgorithm() = default;
  //!
  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::BlockedGibbs;
  }
};

#endif  // BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
