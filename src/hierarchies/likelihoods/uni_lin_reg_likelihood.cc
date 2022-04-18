#include "uni_lin_reg_likelihood.h"

#include "src/utils/eigen_utils.h"

void UniLinRegLikelihood::clear_summary_statistics() {
  mixed_prod = Eigen::VectorXd::Zero(dim);
  data_sum_squares = 0.0;
  covar_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}

double UniLinRegLikelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) const {
  return stan::math::normal_lpdf(
      datum(0), state.regression_coeffs.dot(covariate), sqrt(state.var));
}

void UniLinRegLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                           const Eigen::RowVectorXd &covariate,
                                           bool add) {
  if (add) {
    data_sum_squares += datum(0) * datum(0);
    covar_sum_squares += covariate.transpose() * covariate;
    mixed_prod += datum(0) * covariate.transpose();
  } else {
    data_sum_squares -= datum(0) * datum(0);
    covar_sum_squares -= covariate.transpose() * covariate;
    mixed_prod -= datum(0) * covariate.transpose();
  }
}
