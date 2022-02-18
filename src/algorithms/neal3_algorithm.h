#ifndef BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_

#include "algorithm_id.pb.h"
#include "neal2_algorithm.h"

//! Template class for Neal's algorithm 3 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 3 from Neal (2000)
//! that generates a Markov chain on the clustering of the provided data.
//!
//! This algorithm requires the use of a `ConjugateHierarchy` object.
//! Algorithm 3 from Neal (2000) is almost identical to Algorithm 2, except
//! that conjugacy is further exploied by marginalizing the unique values
//! from the state when updating the cluster allocations, which leads to
//! improved efficiency in terms of effective sample size, but might require
//! longer runtimes.
//! For more information, please refer to the `Neal2Algorithm` class, as well
//! as `BaseAlgorithm` and `MarginalAlgorithm` on which it is based.

class Neal3Algorithm : public Neal2Algorithm {
 public:
  Neal3Algorithm() = default;
  ~Neal3Algorithm() = default;

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::Neal3;
  }

  std::shared_ptr<BaseAlgorithm> clone() const override {
    auto out = std::make_shared<Neal3Algorithm>(*this);
    out->set_mixing(mixing->clone());
    out->set_hierarchy(unique_values[0]->deep_clone());
    return out;
  }

 protected:
  void print_startup_message() const override;

  bool update_hierarchy_params() override { return true; }

  Eigen::VectorXd get_cluster_lpdf(const unsigned int data_idx) const override;
};

#endif  // BAYESMIX_ALGORITHMS_NEAL3_ALGORITHM_H_
