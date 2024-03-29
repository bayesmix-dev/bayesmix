#include "uni_norm_likelihood.h"

double UniNormLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::normal_lpdf(datum(0), state.mean, sqrt(state.var));
}

void UniNormLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                         bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
}

void UniNormLikelihood::clear_summary_statistics() {
  data_sum = 0;
  data_sum_squares = 0;
}
