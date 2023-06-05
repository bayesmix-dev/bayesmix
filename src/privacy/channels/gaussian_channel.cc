#include "gaussian_channel.h"

double GaussianChannel::lpdf(Eigen::VectorXd public_datum,
                             Eigen::VectorXd private_datum) {
  return stan::math::normal_lpdf(public_datum, private_datum, std_dev);
}

Eigen::MatrixXd GaussianChannel::sanitize(Eigen::MatrixXd private_data) {
  Eigen::MatrixXd out = private_data;
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < private_data.rows(); i++) {
    for (int j = 0; j < private_data.cols(); j++) {
      out(i, j) += stan::math::normal_rng(0.0, std_dev, rng);
    }
  }
  return out;
}

Eigen::MatrixXd GaussianChannel::get_candidate_private_data(
    Eigen::MatrixXd sanitized_data) {
  return sanitized_data;
}
