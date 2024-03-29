#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_FA_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_FA_STATE_H_

#include <stan/math/rev.hpp>
#include <tuple>

#include "algorithm_state.pb.h"
#include "base_state.h"
#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

namespace State {

/**
 * State of a Factor Analytic model
 *
 * \f[
 *    Y_i = \Lambda\bm{\eta}_i + \bm{\varepsilon}
 * \f]
 *
 * where \f$ Y_i \f$ is a \f$ p \f$-dimensional vetor, \f$ \bm{\eta}_i \f$ is a
 * \f$ d \f$-dimensional one, \f$ \Lambda \f$ is a \f$ p \times d \f$ matrix
 * and \f$ \bm{\varepsilon} \f$ is an error term with mean zero and diagonal
 * covariance matrix \f$ \psi \f$.
 *
 * For faster likelihood evaluation, we store also the `cov_wood` factor and
 * the log determinant of the matrix \f$ \Lambda \Lambda^T + \psi \f$, see
 * the `compute_wood_chol_and_logdet(...)` function for more details.
 *
 * The unconstrained representation for this state is not implemented.
 */

class FA : public BaseState {
 public:
  Eigen::VectorXd mu, psi;
  Eigen::MatrixXd eta, lambda, cov_wood;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> psi_inverse;
  double cov_logdet;

  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  void set_from_proto(const ProtoState &state_, bool update_card) override {
    if (update_card) {
      card = state_.cardinality();
    }

    mu = bayesmix::to_eigen(state_.fa_state().mu());
    psi = bayesmix::to_eigen(state_.fa_state().psi());
    eta = bayesmix::to_eigen(state_.fa_state().eta());
    lambda = bayesmix::to_eigen(state_.fa_state().lambda());
    psi_inverse = psi.cwiseInverse().asDiagonal();
    compute_wood_factors();
  }

  //! Sets cov_logdet and cov_wood by calling
  //! bayesmix::compute_wood_chol_and_logdet()
  void compute_wood_factors() {
    auto [cov_wood_, cov_logdet_] =
        bayesmix::compute_wood_chol_and_logdet(psi_inverse, lambda);
    cov_logdet = cov_logdet_;
    cov_wood = cov_wood_;
  }

  ProtoState get_as_proto() const override {
    bayesmix::FAState state_;
    bayesmix::to_proto(mu, state_.mutable_mu());
    bayesmix::to_proto(psi, state_.mutable_psi());
    bayesmix::to_proto(eta, state_.mutable_eta());
    bayesmix::to_proto(lambda, state_.mutable_lambda());

    bayesmix::AlgorithmState::ClusterState out;
    out.mutable_fa_state()->CopyFrom(state_);
    return out;
  }
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_FACTOR_ANALYZERS_STATE_H_
