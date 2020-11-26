#ifndef BAYESMIX_MIXINGS_DIRICHLET_MIXING_HPP_
#define BAYESMIX_MIXINGS_DIRICHLET_MIXING_HPP_

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
 public:
  struct State {
    double totalmass;
  };

 protected:
  State state;
  std::shared_ptr<bayesmix::DPPrior> prior;

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
    return card / (n + state.totalmass);
  }

  //! Mass probability for choosing a newly created cluster

  //! \param n_clust Number of clusters
  //! \param n       Total number of data points
  //! \return        Probability value
  double mass_new_cluster(const unsigned int n_clust,
                          const unsigned int n) const override {
    return state.totalmass / (n + state.totalmass);
  }

  void initialize() override;

  void update_state(
      const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
      unsigned int n) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void set_prior(const google::protobuf::Message &prior_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  std::string get_id() const override { return "DP"; }
};

#endif  // BAYESMIX_MIXINGS_DIRICHLET_MIXING_HPP_
