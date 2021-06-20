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

//! Class that represents the Pitman-Yor process mixture model.

//! This class represents a popular mixture model for iterative Gibbs sampling
//! algorithms, called the Pitman-Yor process. See Pitman and Yor (1997) for a
//! formal definition. It has two parameters, strength and discount. It is a
//! generalized version of the Dirichlet process, which has discount = 0 and
//! strength = total mass. In the context of the implemented algorithms, it
//! translates to a mixture that assigns a weight to the creation of a new
//! cluster proportional to their cardinalities, but reduced by the discount
//! factor, while the weight for a newly created cluster is the remaining
//! one counting the total amount as the sample size increased by the strength.
//! See Neal (2000) for a thorough explanation of this model representation.
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

  void initialize() override;

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

  void write_state_to_proto(google::protobuf::Message *out) const override;

  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::PY; }

  virtual bool is_conditional() const { return false; }

  virtual bool is_dependent() const { return false; }

 protected:
  void initialize_state() override;
};

#endif  // BAYESMIX_MIXINGS_PITYOR_MIXING_H_
