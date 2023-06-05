#ifndef BAYESMIX_SRC_PRIVACY_GAUSSIAN_CHANNEL_
#define BAYESMIX_SRC_PRIVACY_GAUSSIAN_CHANNEL_

#include <stan/math/rev.hpp>

#include "base_channel.h"
#include "src/utils/rng.h"

class GaussianChannel : public BasePrivacyChannel {
 protected:
  double std_dev;

 public:
  GaussianChannel() = default;
  ~GaussianChannel() = default;

  GaussianChannel(double sd) : std_dev(sd) {}

  double lpdf(Eigen::VectorXd public_datum,
              Eigen::VectorXd private_datum) override;

  Eigen::MatrixXd sanitize(Eigen::MatrixXd private_data) override;

  Eigen::MatrixXd get_candidate_private_data(
      Eigen::MatrixXd sanitized_data) override;
};

#endif  // BAYESMIX_SRC_PRIVACY_GAUSSIAN_CHANNEL_
