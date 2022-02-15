#ifndef BAYESMIX_ALGORITHMS_NEAL2_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL2_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "algorithm_id.pb.h"
#include "marginal_algorithm.h"
#include "src/hierarchies/base_hierarchy.h"

//! Template class for Neal's algorithm 2 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 2 from Neal (2000)
//! that generates a Markov chain on the clustering of the provided data.
//!
//! This algorithm requires the use of a `ConjugateHierarchy` object.

class Neal2Algorithm : public MarginalAlgorithm {
 public:
  Neal2Algorithm() = default;
  ~Neal2Algorithm() = default;

  bool requires_conjugate_hierarchy() const override { return true; }

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::Neal2;
  }

 protected:
  void print_startup_message() const override;

  void sample_allocations() override;

  void sample_unique_values() override;

  Eigen::VectorXd lpdf_marginal_component(
      const std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const override;

  //! Computes prior component of allocation sampling masses for given datum
  //! @param data_idx Index of the considered data point
  //! @return         Allocation weights for the clusters
  virtual Eigen::VectorXd get_cluster_prior_mass(
      const unsigned int data_idx) const;

  //! Computes likelihood component of alloc. sampling masses for given datum
  //! @param data_idx Index of the considered data point
  //! @return         Allocation weights for the clusters
  virtual Eigen::VectorXd get_cluster_lpdf(const unsigned int data_idx) const;
};

#endif  // BAYESMIX_ALGORITHMS_NEAL2_ALGORITHM_H_
