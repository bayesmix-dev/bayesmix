#include "fa_likelihood.h"

#include "src/utils/distributions.h"

void FALikelihood::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
}

double FALikelihood::compute_lpdf(const Eigen::RowVectorXd& datum) const {
  return bayesmix::multi_normal_lpdf_woodbury_chol(
      datum, state.mu, state.psi_inverse, state.cov_wood, state.cov_logdet);
}

void FALikelihood::update_sum_stats(const Eigen::RowVectorXd& datum,
                                    bool add) {
  if (add) {
    data_sum += datum;
  } else {
    data_sum -= datum;
  }
}
