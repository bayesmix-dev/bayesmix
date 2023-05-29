#include "wavelet_channel.h"

double WaveletChannel::haar_phi(double x) {
  return 1.0 * ((x >= 0) & (x < 1));
}

double WaveletChannel::haar_psi(double x) {
  return haar_phi(x) * ((x < 0.5) + (-1) * (x >= 0.5));
}

Eigen::VectorXd WaveletChannel::eval_haar_basis(double x) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(basis_size);
  out(0) = haar_phi(x);
  int idx = 1;
  for (int j = 0; j <= max_j; j++) {
    for (int l = 0; l < std::pow(2, j); l++) {
      out(idx) =
          std::pow(2.0, 1.0 * j / 2) * haar_psi(std::pow(2.0, j) * x - l);
      idx += 1;
    }
  }
  return out;
}

double WaveletChannel::lpdf(Eigen::VectorXd public_datum,
                            Eigen::VectorXd private_datum) {
  Eigen::VectorXd haar_eval = eval_haar_basis(private_datum[0]);
  return stan::math::double_exponential_lpdf(haar_eval, public_datum, scale);
}

Eigen::MatrixXd WaveletChannel::sanitize(Eigen::MatrixXd private_data) {
  int ndata = private_data.rows();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(ndata, basis_size);
  auto& rng = bayesmix::Rng::Instance().get();

  for (int i = 0; i < private_data.rows(); i++) {
    double x = private_data(i, 0);
    Eigen::VectorXd eval_haar = eval_haar_basis(x);
    for (int j = 0; j < eval_haar.size(); j++) {
      out(i, j) = stan::math::double_exponential_rng(eval_haar(j), scale, rng);
    }
  }
  return out;
}

Eigen::MatrixXd WaveletChannel::get_candidate_private_data(
    Eigen::MatrixXd sanitized_data) {
  Eigen::VectorXd out(sanitized_data.rows());
  auto& rng = bayesmix::Rng::Instance().get();
  if (random_init) {
    for (int i = 0; i < sanitized_data.rows(); i++) {
      out(i) = stan::math::uniform_rng(0.0, 1.0, rng);
    }
    return out;
  }

  for (int i = 0; i < sanitized_data.rows(); i++) {
    auto basis_eval = sanitized_data.row(i).tail(std::pow(2, max_j));
    Eigen::VectorXd::Index max_ind;
    basis_eval.array().abs().maxCoeff(&max_ind);
    int k = max_ind;
    double l = 1.0 * k / std::pow(2.0, max_j);
    double u = 1.0 * (k + 1) / std::pow(2.0, max_j);

    if (basis_eval(max_ind) > 0) {
      u = u - (u - l) / 2.0;
    } else {
      l = l + (u - l) / 2.0;
    }

    out(i) = stan::math::uniform_rng(l, u, rng);
  }
  return out;
}
