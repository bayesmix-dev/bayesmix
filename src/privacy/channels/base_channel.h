#ifndef BAYESMIX_SRC_PRIVACY_BASE_CHANNEL_
#define BAYESMIX_SRC_PRIVACY_BASE_CHANNEL_

class BasePrivacyChannel {
 public:
  BasePrivacyChannel() = default;
  ~BasePrivacyChannel() = default;

  virtual double lpdf(Eigen::VectorXd public_datum,
                      Eigen::VectorXd private_datum) = 0;

  virtual Eigen::MatrixXd sanitize(Eigen::MatrixXd private_data) = 0;

  virtual Eigen::MatrixXd get_candidate_private_data(
      Eigen::MatrixXd sanitized_data) = 0;
};

#endif  // BAYESMIX_SRC_PRIVACY_BASE_CHANNEL_
