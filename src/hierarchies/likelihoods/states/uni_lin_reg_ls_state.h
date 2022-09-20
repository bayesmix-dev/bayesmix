#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_UNI_LIN_REG_LS_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_UNI_LIN_REG_LS_STATE_H_

#include <stan/math/rev.hpp>
#include <tuple>

#include "algorithm_state.pb.h"
#include "base_state.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

namespace State {

//! Returns the constrained parametrization from the
//! unconstrained one, i.e. [a, exp(b)],
//! where `a` is equal to the vector `in` excluding its last
//! element, and `b` is the last element in `in`
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> uni_lin_reg_to_constrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  int N = in.size();
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(N);
  out << in.head(N - 1), stan::math::exp(in(N - 1));
  return out;
}

//! Returns the unconstrained parametrization from the
//! constrained one, i.e. [a, log(b)]
//! where `a` is equal to the vector `in` excluding its last
//! element, and `b` is the last element in `in`
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> uni_lin_reg_to_unconstrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  int N = in.size();
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(N);
  out << in.head(N - 1), stan::math::log(in(N - 1));
  return out;
}

//! Returns the log determinant of the jacobian of the map
//! (x, y) -> (x, log(y)), that is the inverse map of the
//! constrained -> unconstrained representation.
template <typename T>
T uni_lin_reg_log_det_jac(Eigen::Matrix<T, Eigen::Dynamic, 1> constrained) {
  T out = 0;
  int N = constrained.size();
  stan::math::positive_constrain(stan::math::log(constrained(N - 1)), out);
  return out;
}

//! State of a scalar linear regression model with parameters
//! (regression_coeffs, var), where var is the variance of the error term.
//! The unconstrained representation is (regression_coeffs, log(var)).
class UniLinRegLS : public BaseState {
 public:
  Eigen::VectorXd regression_coeffs;
  double var;

  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  Eigen::VectorXd get_unconstrained() const override {
    Eigen::VectorXd temp(regression_coeffs.size() + 1);
    temp << regression_coeffs, var;
    return uni_lin_reg_to_unconstrained(temp);
  }

  void set_from_unconstrained(const Eigen::VectorXd &in) override {
    Eigen::VectorXd temp = uni_lin_reg_to_constrained(in);
    int dim = in.size() - 1;
    regression_coeffs = temp.head(dim);
    var = temp(dim);
  }

  void set_from_proto(const ProtoState &state_, bool update_card) override {
    if (update_card) {
      card = state_.cardinality();
    }
    regression_coeffs =
        bayesmix::to_eigen(state_.lin_reg_uni_ls_state().regression_coeffs());
    var = state_.lin_reg_uni_ls_state().var();
  }

  ProtoState get_as_proto() const override {
    ProtoState state;
    bayesmix::to_proto(
        regression_coeffs,
        state.mutable_lin_reg_uni_ls_state()->mutable_regression_coeffs());
    state.mutable_lin_reg_uni_ls_state()->set_var(var);
    return state;
  }

  double log_det_jac() const override {
    Eigen::VectorXd temp(regression_coeffs.size() + 1);
    temp << regression_coeffs, var;
    return uni_lin_reg_log_det_jac(temp);
  }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_UNI_LIN_REG_LS_STATE_H_
