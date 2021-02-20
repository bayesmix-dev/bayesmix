#ifndef BAYESMIX_MIXINGS_PITYOR_MIXING_H_
#define BAYESMIX_MIXINGS_PITYOR_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>

#include "marginal_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/base_hierarchy.h"

//! Class that represents the Pitman-Yor process mixture model.

//! This class represents a particular mixture model for iterative BNP
//! algorithms, called the Pitman-Yor process. It has two parameters, strength
//! and discount. It is a generalized version of the Dirichlet process, which
//! has discount = 0 and strength = total mass. In terms of the algorithms, it
//! translates to a mixture that assigns a weight to the creation of a new
//! cluster proportional to their cardinalities, but reduced by the discount
//! factor, while the weight for a newly created cluster is the remaining
//! one counting the total amount as the sample size increased by the strength.

class PitYorMixing : public MarginalMixing {
 public:
  struct State {
    double strength, discount;
  };

 protected:
  State state;

  //!
  void create_empty_prior() override { prior.reset(new bayesmix::PYPrior); }
  //!
  std::shared_ptr<bayesmix::PYPrior> cast_prior() {
    return std::dynamic_pointer_cast<bayesmix::PYPrior>(prior);
  }
  //!
  void initialize_state() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~PitYorMixing() = default;
  PitYorMixing() = default;

  std::shared_ptr<BaseMixing> clone() const override {
    return std::make_shared<PitYorMixing>(*this);
  }

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
  void initialize(const unsigned int n_clust = 1) override;
  //!
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      unsigned int n) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::PY; }
};

#endif  // BAYESMIX_MIXINGS_PITYOR_MIXING_H_
