#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_UNI_LS_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_UNI_LS_STATE_H_

#include <memory>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "base_state.h"
#include "src/utils/proto_utils.h"

namespace State {

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> uni_ls_to_constrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(2);
  out << in(0), stan::math::exp(in(1));
  return out;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> uni_ls_to_unconstrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(2);
  out << in(0), stan::math::log(in(1));
  return out;
}

template <typename T>
T uni_ls_log_det_jac(Eigen::Matrix<T, Eigen::Dynamic, 1> constrained) {
  T out = 0;
  stan::math::positive_constrain(stan::math::log(constrained(1)), out);
  return out;
}

//! A univariate location-scale state with parametrization (mean, var)
//! The unconstrained representation corresponds to (mean, log(var))
class UniLS : public BaseState {
 public:
  double mean, var;

  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  Eigen::VectorXd get_unconstrained() override {
    Eigen::VectorXd temp(2);
    temp << mean, var;
    return uni_ls_to_unconstrained(temp);
  }

  void set_from_unconstrained(const Eigen::VectorXd &in) override {
    Eigen::VectorXd temp = uni_ls_to_constrained(in);
    mean = temp(0);
    var = temp(1);
  }

  void set_from_proto(const ProtoState &state_, bool update_card) override {
    if (update_card) {
      card = state_.cardinality();
    }
    mean = state_.uni_ls_state().mean();
    var = state_.uni_ls_state().var();
  }

  ProtoState get_as_proto() const override {
    ProtoState state;
    state.mutable_uni_ls_state()->set_mean(mean);
    state.mutable_uni_ls_state()->set_var(var);
    return state;
  }

  double log_det_jac() override {
    Eigen::VectorXd temp(2);
    temp << mean, var;
    return uni_ls_log_det_jac(temp);
  }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_UNI_LS_STATE_H_
