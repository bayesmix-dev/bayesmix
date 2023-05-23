#include "laplace_channel.h"

double LaplaceChannel::lpdf(Eigen::VectorXd public_datum,
                            Eigen::VectorXd private_datum) {
  return stan::math::double_exponential_lpdf(public_datum, private_datum,
                                             scale);
}

Eigen::MatrixXd LaplaceChannel::sanitize(Eigen::MatrixXd private_data) {
  Eigen::MatrixXd out = private_data;
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < private_data.rows(); i++) {
    for (int j = 0; j < private_data.cols(); j++) {
      out(i, j) += stan::math::double_exponential_rng(0.0, scale, rng);
    }
  }
  return out;
}

Eigen::MatrixXd LaplaceChannel::get_candidate_private_data(
      Eigen::MatrixXd sanitized_data) {
  return sanitized_data;
}
