#ifndef BAYESMIX_ALGORITHMS_ALGORITHM_NEAL2_HPP_
#define BAYESMIX_ALGORITHMS_ALGORITHM_NEAL2_HPP_

#include "algorithm_marginal.hpp"

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

class AlgorithmNeal2 : public AlgorithmMarginal {
 protected:
  // AUXILIARY TOOLS
  //! Computes marginal contribution of a given iteration & cluster
  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<HierarchyBase> temp_hier,
      const Eigen::MatrixXd &grid) override;

  // ALGORITHM FUNCTIONS
  void print_startup_message() const override;
  void initialize() override;
  void sample_allocations() override;
  void sample_unique_values() override;
  //! Empty: this algorithm does not use weights
  void sample_weights() override { return; }
  //! Empty: this algorithm does not update hyperparameters
  void update_hypers() override { return; }

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~AlgorithmNeal2() = default;
  AlgorithmNeal2() = default;

  std::string get_id() const override { return "Neal2"; }
};

#endif  // BAYESMIX_ALGORITHMS_ALGORITHM_NEAL2_HPP_
