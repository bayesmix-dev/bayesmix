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

  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  double mass_existing_cluster(const unsigned int n, const bool log,
                               const bool propto,
                               std::shared_ptr<AbstractHierarchy> hier,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override;

  double mass_new_cluster(const unsigned int n, const bool log,
                          const bool propto, const unsigned int n_clust,
                          const Eigen::RowVectorXd &covariate =
                              Eigen::RowVectorXd(0)) const override;

  void set_state_from_proto(const google::protobuf::Message &state_) override;

  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override; 

  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::PY; }

  bool is_conditional() const override { return false; }

  bool is_dependent() const override { return false; }

 protected:
  void initialize_state() override;
};

#endif  // BAYESMIX_MIXINGS_PITYOR_MIXING_H_
