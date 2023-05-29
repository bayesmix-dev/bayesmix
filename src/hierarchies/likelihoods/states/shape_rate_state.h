#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_SHAPE_RATE_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_SHAPE_RATE_STATE_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "base_state.h"
#include "src/utils/proto_utils.h"

namespace State {

//! Returns the constrained parametrization from the
//! unconstrained one, i.e. [in[0], exp(in[1])]
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> shape_rate_to_constrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(2);
  out << stan::math::exp(in(0)), stan::math::exp(in(1));
  return out;
}

//! Returns the unconstrained parametrization from the
//! constrained one, i.e. [log(in[0]), log(in[1])]
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> shape_rate_to_unconstrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(2);
  out << stan::math::log(in(0)), stan::math::log(in(1));
  return out;
}

//! Returns the log determinant of the jacobian of the map
//! (x, y) -> (log(x), log(y)), that is the inverse map of the
//! constrained -> unconstrained representation.
template <typename T>
T shape_rate_log_det_jac(Eigen::Matrix<T, Eigen::Dynamic, 1> constrained) {
  T out = 0;
  stan::math::positive_constrain(stan::math::log(constrained(0)), out);
  stan::math::positive_constrain(stan::math::log(constrained(1)), out);
  return out;
}

//! A univariate shape-rate state
//! The unconstrained representation corresponds to (log(shape), log(rate))
class ShapeRate : public BaseState {
 public:
  double shape, rate;

  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  Eigen::VectorXd get_unconstrained() const override {
    Eigen::VectorXd temp(2);
    temp << shape, rate;
    return shape_rate_to_unconstrained(temp);
  }

  void set_from_unconstrained(const Eigen::VectorXd &in) override {
    Eigen::VectorXd temp = shape_rate_to_constrained(in);
    shape = temp(0);
    rate = temp(1);
  }

  void set_from_proto(const ProtoState &state_, bool update_card) override {
    if (update_card) {
      card = state_.cardinality();
    }
    shape = state_.sr_state().shape();
    rate = state_.sr_state().rate();
  }

  ProtoState get_as_proto() const override {
    ProtoState state;
    state.mutable_sr_state()->set_shape(shape);
    state.mutable_sr_state()->set_rate(rate);
    return state;
  }

  double log_det_jac() const override {
    Eigen::VectorXd temp(2);
    temp << shape, rate;
    return shape_rate_log_det_jac(temp);
  }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_SHAPE_RATE_STATE_H_
