#ifndef BAYESMIX_ALGORITHMS_NEAL8_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL8_ALGORITHM_H_

#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <vector>

#include "neal2_algorithm.hpp"

//! Template class for Neal's algorithm 8 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 8 that generates a
//! Markov chain on the clustering of the provided data.
//!
//! It is a generalization of Neal's algorithm 2 which works for any
//! hierarchical model, even non-conjugate ones, unlike its predecessor. The
//! main difference is the presence of a fixed number of additional unique
//! values, called the auxiliary blocks, which are constantly updated and from
//! which new clusters choose their unique values. They offset the lack of
//! conjugacy of the model by allowing an estimate of the (uncomputable)
//! marginal density via a weighted mean on these new blocks. Other than this
//! and some minor adjustments in the allocation sampling phase to circumvent
//! non-conjugacy, it is the same as Neal's algorithm 2.

class Neal8Algorithm : public Neal2Algorithm {
 protected:
  //! Number of auxiliary blocks
  unsigned int n_aux = 3;

  //! Vector of auxiliary blocks
  std::vector<std::shared_ptr<BaseHierarchy>> aux_unique_values;

  // AUXILIARY TOOLS
  //! Computes marginal contribution of a given iteration & cluster
  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<BaseHierarchy> temp_hier,
      const Eigen::MatrixXd &grid) override;

  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<DependentHierarchy> temp_hier,
      const Eigen::MatrixXd &grid, const Eigen::MatrixXd &covariates) override;

  // ALGORITHM FUNCTIONS
  void print_startup_message() const override;
  void initialize() override;
  void sample_allocations() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~Neal8Algorithm() = default;
  Neal8Algorithm() = default;

  // GETTERS AND SETTERS
  unsigned int get_n_aux() const { return n_aux; }
  void set_n_aux(const unsigned int n_aux_) {
    assert(n_aux_ > 0);
    n_aux = n_aux_;
  }

  std::string get_id() const override { return "Neal8"; }
};

#endif  // BAYESMIX_ALGORITHMS_NEAL8_ALGORITHM_H_
