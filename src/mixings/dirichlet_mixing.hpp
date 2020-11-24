#ifndef BAYESMIX_MIXINGS_DIRICHLET_MIXING_HPP_
#define BAYESMIX_MIXINGS_DIRICHLET_MIXING_HPP_

#include <cassert>
#include <memory>

#include "../../proto/cpp/mixing_prior.pb.h"
#include "../hierarchies/base_hierarchy.hpp"
#include "base_mixing.hpp"

//! Class that represents the Dirichlet process mixture model.

//! This class represents a particular mixture model for iterative BNP
//! algorithms, called the Dirichlet process. It represents the distribution of
//! a random probability measure that fulfills a certain property involving the
//! Dirichlet distribution. In terms of the algorithms, it translates to a
//! mixture that assigns a weight M, called the total mass parameter, to the
//! creation of a new cluster, and weights of already existing clusters are
//! proportional to their cardinalities.

class DirichletMixing : public BaseMixing {
 protected:
  //! Total mass parameters
  double totalmass = 1.0;
  bayesmix::DPPrior prior;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~DirichletMixing() = default;
  DirichletMixing() = default;

  // PROBABILITIES FUNCTIONS
  //! Mass probability for choosing an already existing cluster

  //! \param card Cardinality of the cluster
  //! \param n    Total number of data points
  //! \return     Probability value
  double mass_existing_cluster(const unsigned int card,
                               const unsigned int n) const override {
    return card / (n + totalmass);
  }

  //! Mass probability for choosing a newly created cluster

  //! \param n_clust Number of clusters
  //! \param n       Total number of data points
  //! \return        Probability value
  double mass_new_cluster(const unsigned int n_clust,
                          const unsigned int n) const override {
    return totalmass / (n + totalmass);
  }

  void update_hypers(
      const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
      unsigned int n) override;

  // GETTERS AND SETTERS
  double get_totalmass() const { return totalmass; }

  void set_prior(const google::protobuf::Message &prior_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "DP"; }
};

#endif  // BAYESMIX_MIXINGS_DIRICHLET_MIXING_HPP_
