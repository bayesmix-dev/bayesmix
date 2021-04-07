#ifndef BAYESMIX_MIXINGS_DIRICHLET_MIXING_H_
#define BAYESMIX_MIXINGS_DIRICHLET_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Class that represents the Dirichlet process mixture model.

//! This class represents a particular mixture model for iterative BNP
//! algorithms, called the Dirichlet process. It represents the distribution of
//! a random probability measure that fulfills a certain property involving the
//! Dirichlet distribution. In terms of the algorithms, it translates to a
//! mixture that assigns a weight M, called the total mass parameter, to the
//! creation of a new cluster, and weights of already existing clusters are
//! proportional to their cardinalities.

namespace Dirichlet {
struct State {
  double totalmass, logtotmass;
};
};  // namespace Dirichlet

class DirichletMixing
    : public BaseMixing<DirichletMixing, Dirichlet::State,
                                bayesmix::DPPrior> {
 protected:
  //!
  void initialize_state() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~DirichletMixing() = default;
  DirichletMixing() = default;
  //!
  virtual bool is_conditional() const { return false; }
  //!
  virtual bool is_dependent() const { return false; }

  // PROBABILITIES FUNCTIONS
  //! Mass probability for choosing an already existing cluster
  double mass_existing_cluster(const unsigned int n, const bool log,
                               const bool propto,
                               std::shared_ptr<AbstractHierarchy> hier,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override;

  //! Mass probability for choosing a newly created cluster
  double mass_new_cluster(const unsigned int n, const bool log,
                          const bool propto, const unsigned int n_clust,
                          const Eigen::RowVectorXd &covariate =
                              Eigen::RowVectorXd(0)) const override;
  //!
  void initialize() override;
  //!
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  // GETTERS AND SETTERS
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::DP; }
};

#endif  // BAYESMIX_MIXINGS_DIRICHLET_MIXING_H_
