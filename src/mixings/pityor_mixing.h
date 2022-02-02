#ifndef BAYESMIX_MIXINGS_PITYOR_MIXING_H_
#define BAYESMIX_MIXINGS_PITYOR_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Class that represents the Pitman-Yor process (PY) in Pitman and Yor (1997).
//! The EPPF induced by the PY depends on a `strength` parameter M and  a
//! `discount` paramter d.
//! Given a clustering of n elements into k clusters, each with cardinality
//! n_j, j=1, ..., k, the EPPF of the PY gives the following probabilities for
//! the cluster membership of the (n+1)-th observation:
//!      p(j-th cluster | ...) \propto (n_j - d)
//!      p(k+1-th cluster | ...) \propto M + k * d
//!
//! When `discount=0`, the EPPF of the PY process coincides with the one of the
//! DP with totalmass = strength.
//! For more information about the class, please refer instead to base classes,
//! `AbstractMixing` and `BaseMixing`.

namespace PitYor {
struct State {
  double strength, discount;
};
};  // namespace PitYor

class PitYorMixing
    : public BaseMixing<PitYorMixing, PitYor::State, bayesmix::PYPrior> {
 public:
  PitYorMixing() = default;
  ~PitYorMixing() = default;

  //! Performs conditional update of state, given allocations and unique values
  //! @param unique_values  A vector of (pointers to) Hierarchy objects
  //! @param allocations    A vector of allocations label
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! MixingState message by adding the appropriate type
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::PY; }

  //! Returns whether the mixing is conditional or marginal
  bool is_conditional() const override { return false; }

 protected:
  //! Returns probability mass for an old cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param hier       `Hierarchy` object representing the cluster
  //! @return           Probability value
  double mass_existing_cluster(const unsigned int n, const bool log,
                               const bool propto,
                               std::shared_ptr<AbstractHierarchy> hier,
                               const unsigned int n_clust) const override;

  //! Returns probability mass for a new cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param n_clust    Current number of clusters
  //! @return           Probability value
  double mass_new_cluster(const unsigned int n, const bool log,
                          const bool propto,
                          const unsigned int n_clust) const override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;
};

#endif  // BAYESMIX_MIXINGS_PITYOR_MIXING_H_
