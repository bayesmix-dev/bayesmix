#include "multi_norm_likelihood.h"

#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

double MultiNormLikelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum) const {
  return bayesmix::multi_normal_prec_lpdf(datum, state.mean, state.prec_chol,
                                          state.prec_logdet);
}

void MultiNormLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                           bool add) {
  // Check if dim is not defined yet (this usually doesn't happen if the
  // hierarchy is initialized)
  if (!dim) set_dim(datum.size());
  // Updates
  if (add) {
    data_sum += datum.transpose();
    data_sum_squares += datum.transpose() * datum;
  } else {
    data_sum -= datum.transpose();
    data_sum_squares -= datum.transpose() * datum;
  }
}

void MultiNormLikelihood::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
  data_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}
