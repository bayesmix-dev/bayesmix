#ifndef BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
#define BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

//! Class that represents the truncated stick-break mixture model.

//! This class represents a truncated stick-break model for a mixing measure.
//! It uses the stick-breaking representation of the Dirichlet process, but
//! rather than having infinitely many components, their number is truncated
//! to an arbitrarily fixed amount, which is fixed at all moments. Its state is
//! composed of the mixing weights in log-scale and the sticks, i.e. the values
//! for the Beta-distributed random variables in the stick-breaking
//! representation.
//! For more information about the class, please refer instead to base classes,
//! `AbstractMixing` and `BaseMixing`.

namespace TruncSB {
struct State {
  Eigen::VectorXd sticks, logweights;
};
};  // namespace TruncSB

class TruncatedSBMixing : public BaseMixing<TruncatedSBMixing, TruncSB::State,
                                            bayesmix::TruncSBPrior> {
 public:
  TruncatedSBMixing() = default;
  ~TruncatedSBMixing() = default;

  void initialize() override;

  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  Eigen::VectorXd get_weights(const bool log, const bool propto,
                              const Eigen::RowVectorXd &covariate =
                                  Eigen::RowVectorXd(0)) const override;

  void set_state_from_proto(const google::protobuf::Message &state_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::TruncSB;
  }

  virtual bool is_conditional() const { return true; }

  bool is_dependent() const override { return false; }

 protected:
  void initialize_state() override;

  //! Returns weights in log-scale computing them from sticks
  Eigen::VectorXd logweights_from_sticks() const;

  //! Returns the prior shape parameters of the Beta-distributed sticks
  Eigen::MatrixXd get_prior_shape_parameters() const;
};

#endif  // BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
