#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_BASE_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_BASE_STATE_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "src/utils/proto_utils.h"

namespace State {

//! Abstract base class for a generic state
//!
//! Given a statistical model with likelihood L(y|tau) and prior p(tau)
//! a State class represents the value of tau at a certain MCMC iteration.
//! In addition, each instance stores the cardinality of the number of
//! observations in the model.
//!
//! State classes inheriting from this one should implement the methods
//! `set_from_proto` and `to_proto`, that are used to deserialzie from
//! (and serialize to) a `bayesmix::AlgorithmState::ClusterState`
//! protocol buffer message.
//!
//! Optionally, each state can have an "unconstrained" representation,
//! where a bijective transformation B is applied to `tau`, so that
//! the image of B(tau) is R^d for some d.
//! This is essential for the default updaters such as `RandomWalkUpdater`
//! and `MalaUpdater` to work, but is not necessary for other model-specific
//! updaters.
//! If such a representation is needed, child classes should also implement
//! `get_unconstrained`, `set_from_unconstrained`, and `log_det_jac`.

class BaseState {
 public:
  int card;

  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  //! Returns the unconstrained representation x = B(tau)
  virtual Eigen::VectorXd get_unconstrained() { return Eigen::VectorXd(0); }

  //! Sets the current state as tau = B^{-1}(in)
  //! @param in  the unconstrained representation of the state
  virtual void set_from_unconstrained(const Eigen::VectorXd &in) {}

  //! Returns the log determinant of the jacobian of B^{-1}
  virtual double log_det_jac() { return -1; }

  //! Sets the current state from a protobuf object
  //! @param state_ a bayesmix::AlgorithmState::ClusterState instance
  //! @param update_card if true, the current cardinality is udpdate
  virtual void set_from_proto(const ProtoState &state_, bool update_card) = 0;

  //! Returns a `bayesmix::AlgorithmState::ClusterState` representig the
  //! current value of the state
  virtual ProtoState get_as_proto() const = 0;

  //! Returns a shared pointer to `bayesmix::AlgorithmState::ClusterState`
  //! representig the current value of the state
  std::shared_ptr<ProtoState> to_proto() const {
    return std::make_shared<ProtoState>(get_as_proto());
  }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_BASE_STATE_H_
