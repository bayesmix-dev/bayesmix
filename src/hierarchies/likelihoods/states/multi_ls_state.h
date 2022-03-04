#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_MULTI_LS_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_MULTI_LS_STATE_H_

#include <stan/math/rev.hpp>
#include <tuple>

#include "algorithm_state.pb.h"
#include "src/utils/proto_utils.h"

namespace State {

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> multi_ls_to_unconstrained(
    Eigen::Matrix<T, Eigen::Dynamic, 1> mean_in,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> prec_in) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> prec_out =
      stan::math::cov_matrix_free(prec_in);
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(mean_in.size() + prec_out.size());
  out << mean_in, prec_out;
  return out;
}

template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, 1>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
multi_ls_to_constrained(Eigen::Matrix<T, Eigen::Dynamic, 1> in) {
  double dim_ = 0.5 * (std::sqrt(8 * in.size() + 9) - 3);
  double dimf;
  assert(modf(dim_, &dimf) == 0.0);
  int dim = int(dimf);
  Eigen::Matrix<T, Eigen::Dynamic, 1> mean(dim);
  mean << in.head(dim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> prec(dim, dim);
  prec = stan::math::cov_matrix_constrain(in.tail(in.size() - dim), dim);
  return std::make_tuple(mean, prec);
}

template <typename T>
T multi_ls_log_det_jac(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> prec_constrained) {
  T out = 0;
  stan::math::positive_constrain(stan::math::cov_matrix_free(prec_constrained),
                                 out);
  return out;
}

class MultiLS {
 public:
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec, prec_chol;
  double prec_logdet;

  Eigen::VectorXd get_unconstrained() {
    return multi_ls_to_unconstrained(mean, prec);
  }

  void set_from_unconstrained(Eigen::VectorXd in) {
    std::tie(mean, prec) = multi_ls_to_constrained(in);
    set_from_constrained(mean, prec);
  }

  void set_from_constrained(Eigen::VectorXd mean_, Eigen::MatrixXd prec_) {
    mean = mean_;
    prec = prec_;
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

  double log_det_jac() { return multi_ls_log_det_jac(prec); }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_MULTI_LS_STATE_H_
