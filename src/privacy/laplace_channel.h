#ifndef BAYESMIX_SRC_PRIVACY_LAPLACE_CHANNEL_
#define BAYESMIX_SRC_PRIVACY_LAPLACE_CHANNEL_

#include <stan/math/rev.hpp>

#include "base_channel.h"
#include "src/utils/rng.h"

class LaplaceChannel : public BasePrivacyChannel {
 protected:
  double scale;

 public:
  LaplaceChannel() = default;
  ~LaplaceChannel() = default;

  LaplaceChannel(double scale_) : scale(scale_) {}

  double lpdf(Eigen::VectorXd public_datum,
              Eigen::VectorXd private_datum) override;

  Eigen::MatrixXd sanitize(Eigen::MatrixXd private_data) override;
};

#endif  // BAYESMIX_SRC_PRIVACY_LAPLACE_CHANNEL_
