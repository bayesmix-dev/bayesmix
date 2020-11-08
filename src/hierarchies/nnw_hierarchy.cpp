#include "nnw_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../../proto/cpp/matrix.pb.h"
#include "../utils/distributions.hpp"
#include "../utils/proto_utils.hpp"
#include "../utils/rng.hpp"

void NNWHierarchy::check_hypers_validity() {
  unsigned int dim = mu0.size();
  assert(lambda > 0);
  assert(dim == tau0.rows());
  assert(nu > dim - 1);

  // Check if tau0 is a square symmetric positive semidefinite matrix
  assert(tau0.rows() == tau0.cols());
  assert(tau0.isApprox(tau0.transpose()));
  Eigen::LLT<Eigen::MatrixXd> llt(tau0);
  assert(llt.info() != Eigen::NumericalIssue);
}

void NNWHierarchy::check_state_validity() {
  // Check if tau is a square matrix
  unsigned int dim = mean.size();
  assert(dim == tau.rows());
  assert(dim == tau.cols());
  // Check if tau is symmetric positive semi definite
  assert(tau.isApprox(tau.transpose()));
  assert(tau_chol_factor.info() != Eigen::NumericalIssue);
}

//! \param tau Value to set to state[1]
void NNWHierarchy::set_tau_and_utilities(const Eigen::MatrixXd &tau_) {
  tau = tau_;

  // Update tau utilities
  tau_chol_factor = Eigen::LLT<Eigen::MatrixXd>(tau);
  tau_chol_factor_eval = tau_chol_factor.matrixL().transpose();
  Eigen::VectorXd diag = tau_chol_factor_eval.diagonal();
  tau_logdet = 2 * log(diag.array()).sum();
}

void NNWHierarchy::check_and_initialize() {
  check_hypers_validity();
  unsigned int dim = get_mu0().size();
  mean = get_mu0();
  set_tau_and_utilities(get_lambda() * Eigen::MatrixXd::Identity(dim, dim));
}

//! \param data                  Matrix of row-vectorial data points
//! \param mu0, lambda, tau0, nu Original values for hyperparameters
//! \return                      Vector of updated values for hyperparameters
std::vector<Eigen::MatrixXd> NNWHierarchy::normal_wishart_update(
    const Eigen::MatrixXd &data, const EigenRowVec &mu0, const double lambda,
    const Eigen::MatrixXd &tau0_inv, const double nu) {
  // Initialize relevant objects
  unsigned int n = data.rows();
  Eigen::MatrixXd lambda_post(1, 1), nu_post(1, 1);

  // Compute updated hyperparameters
  EigenRowVec mubar = data.colwise().mean();  // sample mean
  lambda_post(0, 0) = lambda + n;
  nu_post(0, 0) = nu + 0.5 * n;
  EigenRowVec mu_post = (lambda * mu0 + n * mubar) * (1 / (lambda + n));
  // Compute tau_post
  Eigen::MatrixXd tau_temp = Eigen::MatrixXd::Zero(data.cols(), data.cols());
  for (size_t i = 0; i < n; i++) {
    EigenRowVec datum = data.row(i);
    tau_temp += (datum - mubar).transpose() * (datum - mubar);  // column * row
  }
  tau_temp +=
      (n * lambda / (n + lambda)) * (mubar - mu0).transpose() * (mubar - mu0);
  tau_temp = 0.5 * tau_temp + tau0_inv;
  Eigen::MatrixXd tau_post = stan::math::inverse_spd(tau_temp);
  return std::vector<Eigen::MatrixXd>{mu_post, lambda_post, tau_post, nu_post};
}

//! \param data Matrix of row-vectorial data points
//! \return     Likehood vector evaluated in data
Eigen::VectorXd NNWHierarchy::like(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = lpdf(data);
  return result.array().exp();
}

//! \param data Matrix of row-vectorial data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd NNWHierarchy::lpdf(const Eigen::MatrixXd &data) {
  // Initialize relevant objects
  unsigned int n = data.rows();
  Eigen::VectorXd result(n);
  // Compute likelihood for each data point
  for (size_t i = 0; i < n; i++) {
    result(i) = bayesmix::multi_normal_prec_lpdf(
        data.row(i), mean, tau_chol_factor_eval, tau_logdet);
  }
  return result;
}

//! \param data Matrix of row-vectorial data points
//! \return     Marginal distribution vector evaluated in data
Eigen::VectorXd NNWHierarchy::eval_marg(const Eigen::MatrixXd &data) {
  Eigen::VectorXd result = marg_lpdf(data);
  return result.array().exp();
}

//! \param data Matrix of row-vectorial data points
//! \return     Marginal distribution vector evaluated in data
Eigen::VectorXd NNWHierarchy::marg_lpdf(const Eigen::MatrixXd &data) {
  // Initialize relevant objects
  unsigned int n = data.rows();
  Eigen::VectorXd result(n);
  unsigned int dim = data.cols();

  // Get values of hyperparameters
  EigenRowVec mu0 = get_mu0();
  double lambda = get_lambda();
  Eigen::MatrixXd tau0_inv = get_tau0_inv();
  double nu = get_nu();

  // Compute dof and scale of marginal distribution
  double nu_n = 2 * nu - dim + 1;
  Eigen::MatrixXd sigma_n =
      tau0_inv * (nu - 0.5 * (dim - 1)) * lambda / (lambda + 1);

  for (size_t i = 0; i < n; i++) {
    // Compute marginal for each data point
    EigenRowVec datum = data.row(i);
    result(i) = stan::math::multi_student_t_lpdf(datum, nu_n, mu0, sigma_n);
  }
  return result;
}

void NNWHierarchy::draw() {
  // Get values of hyperparameters
  EigenRowVec mu0 = get_mu0();
  double lambda = get_lambda();
  Eigen::MatrixXd tau0 = get_tau0();
  double nu = get_nu();

  // Generate new state values from their prior centering distribution
  auto rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new = stan::math::wishart_rng(nu, tau0, rng);
  EigenRowVec mean =
      stan::math::multi_normal_prec_rng(mu0, tau_new * lambda, rng);

  // Update state
  set_tau_and_utilities(tau_new);
}

//! \param data Matrix of row-vectorial data points
void NNWHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Get values of hyperparameters
  EigenRowVec mu0 = get_mu0();
  double lambda = get_lambda();
  Eigen::MatrixXd tau0_inv = get_tau0_inv();
  double nu = get_nu();

  // Update values
  std::vector<Eigen::MatrixXd> temp =
      normal_wishart_update(data, mu0, lambda, tau0_inv, nu);
  EigenRowVec mu_post = temp[0];
  double lambda_post = temp[1](0, 0);
  Eigen::MatrixXd tau_post = temp[2];
  double nu_post = temp[3](0, 0);

  // Generate new state values from their prior centering distribution
  auto rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new = stan::math::wishart_rng(nu_post, tau_post, rng);
  EigenRowVec mean =
      stan::math::multi_normal_prec_rng(mu_post, tau_new * lambda_post, rng);

  // Update state
  set_tau_and_utilities(tau_new);
}

void NNWHierarchy::set_state(google::protobuf::Message *curr, bool check) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  MarginalState::ClusterVal *currcast =
      down_cast<MarginalState::ClusterVal *>(curr);

  mean = to_eigen(currcast->multi_ls_state().mean());
  set_tau_and_utilities(to_eigen(currcast->multi_ls_state().precision()));

  if (check) {
    check_state_validity();
  }
}

void NNWHierarchy::write_state_to_proto(google::protobuf::Message *out) {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  MultiLSState state;
  to_proto(mean, state.mutable_mean());
  to_proto(tau, state.mutable_precision());

  down_cast<MarginalState::ClusterVal *>(out)
      ->mutable_multi_ls_state()
      ->CopyFrom(state);
}
