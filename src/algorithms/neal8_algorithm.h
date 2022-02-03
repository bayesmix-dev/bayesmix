#ifndef BAYESMIX_ALGORITHMS_NEAL8_ALGORITHM_H_
#define BAYESMIX_ALGORITHMS_NEAL8_ALGORITHM_H_

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_id.pb.h"
#include "neal2_algorithm.h"

//! Template class for Neal's algorithm 8 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 8 from Neal (2000)
//! that generates a Markov chain on the clustering of the provided data.
//!
//! It extends Neal's algorithm 2 to also deal with cases when the
//! kernel/likelihood f(x | phi) is not conjugate to G, thanks to the
//! introduction of additional, auxiliary unique values.

class Neal8Algorithm : public Neal2Algorithm {
 public:
  // DESTRUCTOR AND CONSTRUCTORS
  Neal8Algorithm() = default;
  ~Neal8Algorithm() = default;

  bool requires_conjugate_hierarchy() const override { return false; }

  //! Returns number of auxiliary blocks
  unsigned int get_n_aux() const { return n_aux; }

  //! Sets number of auxiliary blocks
  void set_n_aux(const unsigned int n_aux_) {
    if (n_aux_ == 0) {
      throw std::invalid_argument("Number of auxiliary block must be > 0");
    }
    n_aux = n_aux_;
  }

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::Neal8;
  }

  void read_params_from_proto(
      const bayesmix::AlgorithmParams &params) override;

  std::shared_ptr<BaseAlgorithm> clone() override {
    auto out = std::make_shared<Neal8Algorithm>(*this);
    out->set_mixing(mixing->clone());
    out->set_mixing(mixing->clone());
    out->set_hierarchy(unique_values[0]->deep_clone());
    return out;
  }

 protected:
  void initialize() override;

  void print_startup_message() const override;

  void sample_allocations() override;

  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
      const Eigen::RowVectorXd &covariate) const override;

  Eigen::VectorXd get_cluster_prior_mass(
      const unsigned int data_idx) const override;

  Eigen::VectorXd get_cluster_lpdf(const unsigned int data_idx) const override;

  //! Number of auxiliary blocks
  unsigned int n_aux = 3;

  //! Vector of auxiliary blocks
  std::vector<std::shared_ptr<AbstractHierarchy>> aux_unique_values;
};

#endif  // BAYESMIX_ALGORITHMS_NEAL8_ALGORITHM_H_
