#ifndef BAYESMIX_SRC_PRIVACY_WAVELET_CHANNEL_
#define BAYESMIX_SRC_PRIVACY_WAVELET_CHANNEL_

#include <stan/math/rev.hpp>

#include "base_channel.h"
#include "src/utils/rng.h"

/*
 * Privacy Channel as in Butucea et al (2020, Bernoulli) based on a
 * Wavelet Basis. It assumes that data are supported on [0, 1].
 * For the moment only the Haar Basis is considered.
 */
class WaveletChannel : public BasePrivacyChannel {
 protected:
  double scale;
  int max_j;
  int basis_size;

  Eigen::VectorXd j_range;

  bool random_init;

 public:
  WaveletChannel() = default;
  ~WaveletChannel() = default;

  WaveletChannel(int max_j, double lap_scale, bool random_init = false)
      : max_j(max_j), scale(lap_scale), random_init(random_init) {
    basis_size = 1;
    for (int j = 0; j <= max_j; j++) {
      basis_size += std::pow(2, j);
    }
  }

  Eigen::VectorXd eval_haar_basis(double x);

  double haar_phi(double x);

  double haar_psi(double x);

  double lpdf(Eigen::VectorXd public_datum,
              Eigen::VectorXd private_datum) override;

  Eigen::MatrixXd sanitize(Eigen::MatrixXd private_data) override;

  int get_basis_size() { return basis_size; }

  Eigen::MatrixXd get_candidate_private_data(
      Eigen::MatrixXd sanitized_data) override;
};

#endif  // BAYESMIX_SRC_PRIVACY_WAVELET_CHANNEL_
