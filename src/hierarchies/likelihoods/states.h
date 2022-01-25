#ifndef BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_

#include <Eigen/Dense>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"

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

struct MultiLS {
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec, prec_chol;
  double prec_logdet;
};

struct UniLinReg {
  Eigen::VectorXd regression_coeffs;
  double var;
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
