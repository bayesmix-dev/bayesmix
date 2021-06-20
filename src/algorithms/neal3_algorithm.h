#ifndef BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "neal2_algorithm.h"

//! Template class for Neal's algorithm 3 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 3 from Neal (2000)
//! that generates a Markov chain on the clustering of the provided data.
//!
//! This algorithm is almost identical to its algorithm 2 counterpart, except
//! that the weights in the allocation sampling phase use the conditional
//! predictive distribution of the cluster given its current observations,
//! rather than the likelihood given its own unique values.
//! For more information, please refer to the `Neal2Algorithm` class, as well
//! as `BaseAlgorithm` and `MarginalAlgorithm` on which it is based.

class Neal3Algorithm : public Neal2Algorithm {
 public:
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
