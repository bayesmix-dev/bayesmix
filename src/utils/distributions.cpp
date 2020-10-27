#include "distributions.hpp"

int bayesmix::categorical_rng(Eigen::VectorXd probas, std::mt19937_64 rng,
                              int start /*= 0*/) {
  return stan::math::categorical_rng(probas, rng) - (start + 1);
}

double bayesmix::multi_normal_prec_lpdf(Eigen::VectorXd datum,
                                        Eigen::VectorXd mean,
                                        Eigen::MatrixXd prec_chol,
                                        double prec_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  double base = prec_logdet + NEG_LOG_SQRT_TWO_PI * datum.size();
  double exp = (prec_chol * (datum - mean)).squaredNorm();
  return 0.5 * (base - exp);
}
