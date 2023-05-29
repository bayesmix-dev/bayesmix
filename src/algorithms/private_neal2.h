#ifndef BAYESMIX_ALGORITHMS_PRIVATE_NEAL2_H_
#define BAYESMIX_ALGORITHMS_PRIVATE_NEAL2_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "algorithm_id.pb.h"
#include "marginal_algorithm.h"
#include "neal2_algorithm.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/privacy/base_channel.h"
#include "src/utils/rng.h"

//! Template class for Neal's algorithm 2 for conjugate hierarchies

//! This class implements Neal's Gibbs sampling algorithm 2 from Neal (2000)
//! that generates a Markov chain on the clustering of the provided data.
//!
//! This algorithm requires the use of a `ConjugateHierarchy` object.

class PrivateNeal2 : public MarginalAlgorithm {
 protected:
  Eigen::MatrixXd public_data;
  Eigen::MatrixXd& private_data = data;
  int n_acc = 0;
  int n_prop = 0;
  std::shared_ptr<BasePrivacyChannel> privacy_channel;

 public:
  PrivateNeal2() = default;
  ~PrivateNeal2() = default;

  bool requires_conjugate_hierarchy() const override { return false; }

  void set_channel(const std::shared_ptr<BasePrivacyChannel> channel) {
    privacy_channel = channel;
  }

  bayesmix::AlgorithmId get_id() const override {
    return bayesmix::AlgorithmId::PrivateNeal2;
  }

  std::shared_ptr<BaseAlgorithm> clone() const override {
    auto out = std::make_shared<PrivateNeal2>(*this);
    out->set_mixing(mixing->clone());
    out->set_hierarchy(unique_values[0]->deep_clone());
    return out;
  }

  double get_acceptance_rate() { return (1.0 * n_acc) / n_prop; }

  void set_public_data(const Eigen::MatrixXd& public_data_);

 protected:
  void print_startup_message() const override;

  void sample_allocations() override;

  void sample_unique_values() override;

  Eigen::VectorXd lpdf_marginal_component(
      const std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::MatrixXd& grid,
      const Eigen::RowVectorXd& covariate) const override;

  //! Computes prior component of allocation sampling masses for given datum
  //! @param data_idx Index of the considered data point
  //! @return         Allocation weights for the clusters
  Eigen::VectorXd get_cluster_prior_mass(const unsigned int data_idx) const;
};

#endif  // BAYESMIX_ALGORITHMS_PRIVATE_NEAL2_H_