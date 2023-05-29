#include "beta_likelihood.h"

double BetaLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::beta_lpdf(datum(0), state.shape, state.rate);
}

Eigen::VectorXd BetaLikelihood::sample() const {
  Eigen::VectorXd out(1);
  auto &rng = bayesmix::Rng::Instance().get();
  out(0) = stan::math::beta_rng(state.shape, state.rate, rng);
  return out;
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
