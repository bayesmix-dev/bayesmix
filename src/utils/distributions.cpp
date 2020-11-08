#include "distributions.hpp"
#include <Eigen/Dense>
#include <random>
#include <stan/math/prim/prob.hpp>

int bayesmix::categorical_rng(const Eigen::VectorXd &probas,
                              std::mt19937_64 &rng, int start /*= 0*/) {
  return stan::math::categorical_rng(probas, rng) - (start + 1);
}

double bayesmix::multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                                        const Eigen::VectorXd &mean,
                                        const Eigen::MatrixXd &prec_chol,
                                        double prec_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  double base = prec_logdet + NEG_LOG_SQRT_TWO_PI * datum.size();
  double exp = (prec_chol * (datum - mean)).squaredNorm();
  return 0.5 * (base - exp);
}
