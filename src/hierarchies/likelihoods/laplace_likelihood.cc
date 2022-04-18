#include "laplace_likelihood.h"

double LaplaceLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::double_exponential_lpdf(
      datum(0), state.mean, stan::math::sqrt(state.var / 2.0));
}
