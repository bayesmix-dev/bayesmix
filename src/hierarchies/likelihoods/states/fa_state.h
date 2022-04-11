#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_FA_STATE_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_FA_STATE_H_

#include <stan/math/rev.hpp>
#include <tuple>

#include "algorithm_state.pb.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

namespace State {

class FA {
 public:
  Eigen::VectorXd mu, psi;
  Eigen::MatrixXd eta, lambda, cov_wood;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> psi_inverse;
  double cov_logdet;
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_STATES_FACTOR_ANALYZERS_STATE_H_
