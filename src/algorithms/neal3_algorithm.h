#ifndef BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "neal2_algorithm.h"

//! TODO class description

class Neal3Algorithm : public Neal2Algorithm {
 public:
  // DESTRUCTOR AND CONSTRUCTORS
  Neal3Algorithm() = default;
  ~Neal3Algorithm() = default;

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::Neal3;
  }

 protected:
  void print_startup_message() const override;

  bool update_hierarchy_params() override { return true; }

  Eigen::VectorXd get_cluster_lpdf(const unsigned int data_idx) const override;
};

#endif  // BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_
