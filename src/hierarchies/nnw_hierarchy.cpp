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
  assert(lambda0 > 0);
  assert(dim == tau0.rows());
  assert(nu0 > dim - 1);

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
  set_tau_and_utilities(get_lambda0() * Eigen::MatrixXd::Identity(dim, dim));
}

//! \param data                    Matrix of row-vectorial data points
//! \param mu0, lambda0, tau0, nu0 Original values for hyperparameters
//! \return                        Vector of updated values for hyperparameters
NNWHierarchy::PostParams NNWHierarchy::normal_wishart_update(
    const Eigen::MatrixXd &data, const Eigen::RowVectorXd &mu0,
    const double lambda0, const Eigen::MatrixXd &tau0_inv, const double nu0) {
  // Initialize relevant objects
  PostParams out;

  unsigned int n = data.rows();
  // Eigen::MatrixXd lambda_n(1, 1), nu_n(1, 1);

  // Compute updated hyperparameters
  out.lambda_n = lambda0 + n;
  out.nu_n = nu0 + 0.5 * n;

  Eigen::RowVectorXd mubar = data.colwise().mean();  // sample mean
  out.mu_n = (lambda0 * mu0 + n * mubar) / (lambda0 + n);
  // Compute tau_n
  Eigen::MatrixXd tau_temp = Eigen::MatrixXd::Zero(data.cols(), data.cols());
  for (size_t i = 0; i < n; i++) {
    Eigen::RowVectorXd datum = data.row(i);
    tau_temp += (datum - mubar).transpose() * (datum - mubar);  // column * row
  }
  tau_temp += (n * lambda0 / (n + lambda0)) * (mubar - mu0).transpose() *
              (mubar - mu0);
  tau_temp = 0.5 * tau_temp + tau0_inv;
  out.tau_n = stan::math::inverse_spd(tau_temp);
  return out;
}

//! \param data Matrix of row-vectorial single data point
//! \return     Log-Likehood vector evaluated in data
double NNWHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  // Initialize relevant objects
  return bayesmix::multi_normal_prec_lpdf(datum, mean, tau_chol_factor_eval,
                                          tau_logdet);
}

//! \param data Matrix of row-vectorial data points
//! \return     Log-Likehood vector evaluated in data
Eigen::VectorXd NNWHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data) const {
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

//! \param data Matrix of row-vectorial a single data point
//! \return     Marginal distribution vector evaluated in data
double NNWHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) const {
  unsigned int dim = datum.cols();

  // Compute dof and scale of marginal distribution
  double nu_n = 2 * nu0 - dim + 1;
  Eigen::MatrixXd sigma_n =
      tau0_inv * (nu0 - 0.5 * (dim - 1)) * lambda0 / (lambda0 + 1);

  // TODO: chec if this is optimized as our bayesmix::multi_normal_prec_lpdf
  return stan::math::multi_student_t_lpdf(datum, nu_n, mu0, sigma_n);
}

//! \param data Matrix of row-vectorial data points
//! \return     Marginal distribution vector evaluated in data
Eigen::VectorXd NNWHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data) const {
  // Initialize relevant objects
  unsigned int n = data.rows();
  Eigen::VectorXd result(n);
  unsigned int dim = data.cols();

  // Compute dof and scale of marginal distribution
  double nu_n = 2 * nu0 - dim + 1;
  Eigen::MatrixXd sigma_n =
      tau0_inv * (nu0 - 0.5 * (dim - 1)) * lambda0 / (lambda0 + 1);

  for (size_t i = 0; i < n; i++) {
    // Compute marginal for each data point
    Eigen::RowVectorXd datum = data.row(i);
    result(i) = stan::math::multi_student_t_lpdf(datum, nu_n, mu0, sigma_n);
  }
  return result;
}

void NNWHierarchy::draw() {
  // Get values of hyperparameters
  Eigen::RowVectorXd mu0 = get_mu0();
  double lambda0 = get_lambda0();
  Eigen::MatrixXd tau0 = get_tau0();
  double nu0 = get_nu0();

  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new = stan::math::wishart_rng(nu0, tau0, rng);
  Eigen::RowVectorXd mean =
      stan::math::multi_normal_prec_rng(mu0, tau_new * lambda0, rng);

  // Update state
  set_tau_and_utilities(tau_new);
}

//! \param data Matrix of row-vectorial data points
void NNWHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Get values of hyperparameters
  Eigen::RowVectorXd mu0 = get_mu0();
  double lambda0 = get_lambda0();
  Eigen::MatrixXd tau0_inv = get_tau0_inv();
  double nu0 = get_nu0();

  // Update values
  PostParams params = normal_wishart_update(data, mu0, lambda0, tau0_inv, nu0);

  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(params.nu_n, params.tau_n, rng);
  Eigen::RowVectorXd mean = stan::math::multi_normal_prec_rng(
      params.mu_n, tau_new * params.lambda_n, rng);

  // Update state
  set_tau_and_utilities(tau_new);
}

void NNWHierarchy::set_state(const google::protobuf::Message &state_,
                             bool check /*= true*/) {
  const bayesmix::MarginalState::ClusterVal &currcast =
      google::protobuf::internal::down_cast<
          const bayesmix::MarginalState::ClusterVal &>(state_);

  mean = to_eigen(currcast.multi_ls_state().mean());
  set_tau_and_utilities(to_eigen(currcast.multi_ls_state().precision()));

  if (check) {
    check_state_validity();
  }
}

void NNWHierarchy::write_state_to_proto(google::protobuf::Message *out) const {
  using namespace google::protobuf::internal;
  using namespace bayesmix;

  MultiLSState state;
  to_proto(mean, state.mutable_mean());
  to_proto(tau, state.mutable_precision());

  down_cast<MarginalState::ClusterVal *>(out)
      ->mutable_multi_ls_state()
      ->CopyFrom(state);
}
