#ifndef BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_

#include <Eigen/Dense>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "src/utils/proto_utils.h"

namespace State {

class UniLS {
 public:
  double mean, var;

  Eigen::VectorXd get_unconstrained() {
    Eigen::VectorXd out(2);
    out << mean, std::log(var);
    return out;
  }

  void set_from_unconstrained(Eigen::VectorXd in) {
    mean = in(0);
    var = std::exp(in(1));
  }

  void set_from_proto(const bayesmix::AlgorithmState::ClusterState &state_) {
    mean = state_.uni_ls_state().mean();
    var = state_.uni_ls_state().var();
  }

  bayesmix::AlgorithmState::ClusterState get_as_proto() {
    bayesmix::AlgorithmState::ClusterState state;
    state.mutable_uni_ls_state()->set_mean(mean);
    state.mutable_uni_ls_state()->set_var(var);
    return state;
  }

  double log_det_jac() {
    double out = 0;
    stan::math::positive_constrain(std::log(var), out);
    return out;
  }
};

class MultiLS {
 public:
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec, prec_chol;
  double prec_logdet;

  Eigen::VectorXd get_unconstrained() {
    Eigen::VectorXd out_prec = stan::math::cov_matrix_free(prec);
    Eigen::VectorXd out(mean.size() + out_prec.size());
    out << mean, out_prec;
    return out;
  }

  void set_from_unconstrained(Eigen::VectorXd in) {
    double dim_ = 0.5 * (std::sqrt(8 * in.size() + 9) - 3);
    double dim;
    assert(modf(dim_, &dim) == 0.0);
    mean = in.head(int(dim));
    prec =
        stan::math::cov_matrix_constrain(in.tail(int(in.size() - dim)), dim);
    prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec).matrixL();
    Eigen::VectorXd diag = prec_chol.diagonal();
    prec_logdet = 2 * log(diag.array()).sum();
  }

  void set_from_proto(const bayesmix::AlgorithmState::ClusterState &state_) {
    mean = to_eigen(state_.multi_ls_state().mean());
    prec = to_eigen(state_.multi_ls_state().prec());
    prec_chol = to_eigen(state_.multi_ls_state().prec_chol());
    Eigen::VectorXd diag = prec_chol.diagonal();
    prec_logdet = 2 * log(diag.array()).sum();
  }

  bayesmix::AlgorithmState::ClusterState get_as_proto() {
    bayesmix::AlgorithmState::ClusterState state;
    bayesmix::to_proto(mean, state.mutable_multi_ls_state()->mutable_mean());
    bayesmix::to_proto(prec, state.mutable_multi_ls_state()->mutable_prec());
    bayesmix::to_proto(prec_chol,
                       state.mutable_multi_ls_state()->mutable_prec_chol());
    return state;
  }

  double log_det_jac() {
    double out = 0;
    stan::math::positive_constrain(stan::math::cov_matrix_free(prec), out);
    return out;
  }
};

struct UniLinReg {
  Eigen::VectorXd regression_coeffs;
  double var;
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
