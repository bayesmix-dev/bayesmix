#ifndef BAYESMIX_ALGORITHMS_NEAL2_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL2_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>

#include "algorithm_id.pb.h"
#include "marginal_algorithm.h"
#include "src/hierarchies/base_hierarchy.h"

//! Template class for Neal's algorithm 2 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 2 that generates a
//! Markov chain on the clustering of the provided data.
//!
//! Using this algorithm implicitly assumes that the provided hierarchy class
//! represents a conjugate model, i.e. in which posterior distributions have
//! the same form as their corresponding prior distributions. Conjugacy is made
//! use of in the computation of the estimated density's marginal component,
//! since the marginal distribution for the data can be expressed analytically.
//!
//! The basic idea for this algorithm is randomly drawing new allocations for
//! data points according to weights that depend on the cardinalities of the
//! current clustering and on the mixture model used. This way, sometimes new
//! clusters are created, and thus new unique values for them must be generated
//! from the prior centering distribution. After that, unique values for each
//! cluster are instead updated via the posterior distribution, which again has
//! a closed-form expression thanks to conjugacy.

class Neal2Algorithm : public MarginalAlgorithm {
 protected:
  // AUXILIARY TOOLS
  //! Computes marginal contribution of a given iteration & cluster
  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
      const Eigen::MatrixXd &covariates) const override;
  //!
  virtual Eigen::VectorXd get_cluster_prior_mass(
      const unsigned int data_idx) const;
  //!
  virtual Eigen::VectorXd get_cluster_lpdf(const unsigned int data_idx) const;

  // ALGORITHM FUNCTIONS
  //!
  void print_startup_message() const override;
  //!
  void sample_allocations() override;
  //!
  void sample_unique_values() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~Neal2Algorithm() = default;
  Neal2Algorithm() = default;
  //!
  bool requires_conjugate_hierarchy() const override { return true; }
  //!
  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::Neal2;
  }
};

#endif  // BAYESMIX_ALGORITHMS_NEAL2_ALGORITHM_H_
