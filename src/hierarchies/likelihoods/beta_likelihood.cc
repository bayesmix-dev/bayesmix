#include "beta_likelihood.h"

double BetaLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::beta_lpdf(datum(0), state.shape, state.rate);
}

void BetaLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                      bool add) {
  double x = datum(0);
  if (add) {
    sum_logs += std::log(x);
    sum_logs1m += std::log(1. - x);
  } else {
    sum_logs -= std::log(x);
    sum_logs1m -= std::log(1. - x);
  }
}

void BetaLikelihood::clear_summary_statistics() {
  sum_logs = 0;
  sum_logs1m = 0;
}
