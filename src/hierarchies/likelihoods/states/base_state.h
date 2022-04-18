#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_BASE_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_BASE_STATE_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "src/utils/proto_utils.h"

namespace State {

class BaseState {
 public:
  int card;

  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  virtual Eigen::VectorXd get_unconstrained() { return Eigen::VectorXd(0); }

  virtual void set_from_unconstrained(Eigen::VectorXd in) {}

  virtual void set_from_proto(const ProtoState &state_, bool update_card) = 0;

  virtual ProtoState get_as_proto() const = 0;

  std::shared_ptr<ProtoState> to_proto() {
    return std::make_shared<ProtoState>(get_as_proto());
  }

  virtual double log_det_jac() { return -1; }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_BASE_STATE_H_
