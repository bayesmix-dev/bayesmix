#include "nnig_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"

void NNIGHierarchy::check_and_initialize() {
  check_hypers_validity();
  mean = mu0;
  sd = sqrt(beta0 / (alpha0 - 1));
}

//! \param data                        Column vector of data points
//! \param mu0, alpha0, beta0, lambda0 Original values for hyperparameters
//! \return                            Vector of updated values for hyperpar.s
NNIGHierarchy::PostParams NNIGHierarchy::normal_invgamma_update(
    const Eigen::VectorXd &data, const double mu0, const double alpha0,
    const double beta0, const double lambda0) {
  // Initialize relevant variables
  PostParams out;

  unsigned int n = data.rows();

  if (n == 0) {  // no update possible
    out.mu_n = mu0;
    out.alpha_n = alpha0;
    out.beta_n = beta0;
    out.lambda_n = lambda0;
    return out;
  }

  // Compute updated hyperparameters
  double y_bar = data.mean();  // sample mean
  out.mu_n = (lambda0 * mu0 + n * y_bar) / (lambda0 + n);
  out.alpha_n = alpha0 + 0.5 * n;
  double ss = (data.dot(data)) - n * y_bar * y_bar;  // sum of squares
  out.beta_n =
      beta0 + 0.5 * ss +
      0.5 * lambda0 * n * (y_bar - mu0) * (y_bar - mu0) / (n + lambda0);
  out.lambda_n = lambda0 + n;

  return out;
}

//! \param data Column vector of data points
//! \return     Likehood vector evaluated in data
Eigen::VectorXd NNIGHierarchy::like(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = lpdf_grid(data);
  return result.array().exp();
}

//! \param data Column vector containing a single data point
//! \return     Log-Likehood vector evaluated in data
double NNIGHierarchy::lpdf(const Eigen::RowVectorXd &datum) {
  assert(datum.size() == 1);
  return stan::math::normal_lpdf(datum(0), mean, sd);
}

//! \param data Column vector of data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd NNIGHierarchy::lpdf_grid(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute likelihood for each data point
    result(i) = stan::math::normal_lpdf(data(i, 0), mean, sd);
  }
  return result;
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data
Eigen::VectorXd NNIGHierarchy::eval_marg(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = marg_lpdf_grid(data);
  return result.array().exp();
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
double NNIGHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) {
  assert(datum.size() == 1);

  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(beta0 * (lambda0 + 1) / (alpha0 * lambda0));
  return stan::math::student_t_lpdf(datum(0), 2 * alpha0, mu0, sig_n);
}

//! \param data Column vector of data points
//! \return     Marginal distribution vector evaluated in data (log)
Eigen::VectorXd NNIGHierarchy::marg_lpdf_grid(const Eigen::MatrixXd &data) {
  // Compute standard deviation of marginal distribution
  double sig_n = sqrt(beta0 * (lambda0 + 1) / (alpha0 * lambda0));

  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    // Compute marginal for each data point
    result(i) = stan::math::student_t_lpdf(data(i, 0), 2 * alpha0, mu0, sig_n);
  }
  return result;
}

void NNIGHierarchy::draw() {
  // Update state values from their prior centering distribution
  auto& rng = bayesmix::Rng::Instance().get();
  sd = sqrt(stan::math::inv_gamma_rng(alpha0, beta0, rng));
  mean = stan::math::normal_rng(mu0, sd / sqrt(lambda0), rng);
}

//! \param data Column vector of data points
void NNIGHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // std::cout << "sampling given: " << data.transpose() << std::endl;
  // Update values
  PostParams params =
      normal_invgamma_update(data.col(0), mu0, alpha0, beta0, lambda0);

  // Update state values from their prior centering distribution
  auto& rng = bayesmix::Rng::Instance().get();
  sd = sqrt(stan::math::inv_gamma_rng(params.alpha_n, params.beta_n, rng));
  mean = stan::math::normal_rng(params.mu_n, sd / sqrt(params.lambda_n), rng);
}

void NNIGHierarchy::set_state(google::protobuf::Message *curr, bool check) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  MarginalState::ClusterVal *currcast =
      down_cast<MarginalState::ClusterVal *>(curr);

  mean = currcast->univ_ls_state().mean();
  sd = currcast->univ_ls_state().sd();

  if (check) {
    check_state_validity();
  }
}

void NNIGHierarchy::write_state_to_proto(google::protobuf::Message *out) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  UnivLSState state;
  state.set_mean(mean);
  state.set_sd(sd);

  down_cast<MarginalState::ClusterVal *>(out)
      ->mutable_univ_ls_state()
      ->CopyFrom(state);
}
