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

//! Class that represents a truncated stick-breaking process, as shown in
//! Ishwaran and James (2001).
//!
//! A truncated stick-breaking process is a prior for weights (w_1,...,w_H) in
//! the H-1 dimensional unit simplex, and is defined as follows:
//!   w_1 = v_1
//!   w_j = v_j (1 - v_1) ... (1 - v_{j-1}), for j=1, ... H-1
//!   w_H = 1 - (w_1 + w_2 + ... + w_{H-1})
//! The v_j's are called sticks and we assume them to be independently
//! distributed as v_j ~ Beta(a_j, b_j).
//!
//! When a_j = 1 and b_j = M, the stick-breaking process is a truncation of the
//! stick-breaking representation of the DP.
//! When a_j = 1 - d and b_j = M + i*d, it is the trunctation of a PY process.
//! Its state is composed of the weights w_j in log-scale and the sticks v_j.
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
  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::TruncSB;
  }

  //! Returns whether the mixing is conditional or marginal
  bool is_conditional() const override { return true; }

 protected:
  //! Initializes state parameters to appropriate values
  void initialize_state() override;

  //! Returns mixing weights (for conditional mixings only)
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @return           The vector of mixing weights
  Eigen::VectorXd mixing_weights(const bool log,
                                 const bool propto) const override;

  //! Returns weights in log-scale computing them from sticks
  Eigen::VectorXd logweights_from_sticks() const;

  //! Returns the prior shape parameters of the Beta-distributed sticks
  Eigen::MatrixXd get_prior_shape_parameters() const;
};

#endif  // BAYESMIX_MIXINGS_TRUNCATED_SB_MIXING_H_
