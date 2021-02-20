#ifndef BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "conditional_algorithm.h"

class BlockedGibbsAlgorithm : public ConditionalAlgorithm {
  //!
  bool update_hierarchy_params() override { return true; }  // TODO ?
  //!
  void print_startup_message() const override;

 public:
  ~ConditionalAlgorithm() = default;
  ConditionalAlgorithm() = default;
  //!
  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::BlockedGibbs;
  }
};

#endif  // BAYESMIX_ALGORITHMS_BLOCKED_GIBBS_ALGORITHM_H_
