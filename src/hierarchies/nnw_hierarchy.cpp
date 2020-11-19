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
  unsigned int dim = hypers->mu.size();
  assert(hypers->lambda > 0);
  assert(dim == hypers->tau.rows());
  assert(hypers->nu > dim - 1);

  // Check if tau0 is a square symmetric positive semidefinite matrix
  assert(hypers->tau.rows() == hypers->tau.cols());
  assert(hypers->tau.isApprox(hypers->tau.transpose()));
  Eigen::LLT<Eigen::MatrixXd> llt(hypers->tau);
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
  unsigned int dim = hypers->mu.size();
  mean = hypers->mu;
  set_tau_and_utilities(hypers->lambda * Eigen::MatrixXd::Identity(dim, dim));
}

//! \param data                    Matrix of row-vectorial data points
//! \param mu0, lambda0, tau0, nu0 Original values for hyperparameters
//! \return                        Vector of updated values for hyperparameters
NNWHierarchy::Hyperparams NNWHierarchy::normal_wishart_update(
    const Eigen::MatrixXd &data, const Eigen::RowVectorXd &mu0,
    const double lambda0, const Eigen::MatrixXd &tau0_inv, const double nu0) {
  // Initialize relevant objects
  Hyperparams post_params;
  unsigned int n = data.rows();

  // Compute updated hyperparameters
  post_params.lambda = lambda0 + n;
  post_params.nu = nu0 + 0.5 * n;

  Eigen::RowVectorXd mubar = data.colwise().mean();  // sample mean
  post_params.mu = (lambda0 * mu0 + n * mubar) / (lambda0 + n);
  // Compute tau_n
  Eigen::MatrixXd tau_temp = Eigen::MatrixXd::Zero(data.cols(), data.cols());
  for (size_t i = 0; i < n; i++) {
    Eigen::RowVectorXd datum = data.row(i);
    tau_temp += (datum - mubar).transpose() * (datum - mubar);  // column * row
  }
  tau_temp += (n * lambda0 / (n + lambda0)) * (mubar - mu0).transpose() *
              (mubar - mu0);
  tau_temp = 0.5 * tau_temp + tau0_inv;
  post_params.tau = stan::math::inverse_spd(tau_temp);
  return post_params;
}

void NNWHierarchy::update_hypers(
    const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
    unsigned int n) {
  return;
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
  double nu_n = 2 * hypers->nu - dim + 1;
  Eigen::MatrixXd sigma_n = tau0_inv * (hypers->nu - 0.5 * (dim - 1)) *
                            hypers->lambda / (hypers->lambda + 1);

  // TODO: chec if this is optimized as our bayesmix::multi_normal_prec_lpdf
  return stan::math::multi_student_t_lpdf(datum, nu_n, hypers->mu, sigma_n);
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
  double nu_n = 2 * hypers->nu - dim + 1;
  Eigen::MatrixXd sigma_n = tau0_inv * (hypers->nu - 0.5 * (dim - 1)) *
                            hypers->lambda / (hypers->lambda + 1);

  for (size_t i = 0; i < n; i++) {
    // Compute marginal for each data point
    Eigen::RowVectorXd datum = data.row(i);
    result(i) =
        stan::math::multi_student_t_lpdf(datum, nu_n, hypers->mu, sigma_n);
  }
  return result;
}

void NNWHierarchy::draw() {
  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(hypers->nu, hypers->tau, rng);

  // Update state
  Eigen::RowVectorXd mean = stan::math::multi_normal_prec_rng(
      hypers->mu, tau_new * hypers->lambda, rng);
  set_tau_and_utilities(tau_new);
}

//! \param data Matrix of row-vectorial data points
void NNWHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  // Update values
  Hyperparams params = normal_wishart_update(data, hypers->mu, hypers->lambda,
                                             tau0_inv, hypers->nu);

  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(params.nu, params.tau, rng);
  Eigen::RowVectorXd mean = stan::math::multi_normal_prec_rng(
      params.mu, tau_new * params.lambda, rng);

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

void NNWHierarchy::set_prior(const google::protobuf::Message &prior_) {
  const bayesmix::NNWPrior &currcast =
      google::protobuf::internal::down_cast<const bayesmix::NNWPrior &>(
          prior_);
  prior = currcast;
  hypers = std::make_shared<Hyperparams>();
  if (prior.has_fixed_values()) {
    hypers->mu = bayesmix::to_eigen(prior.fixed_values().mu0());
    hypers->lambda = prior.fixed_values().lambda0();
    hypers->tau = bayesmix::to_eigen(prior.fixed_values().tau0());
    tau0_inv = stan::math::inverse_spd(hypers->tau);
    hypers->nu = prior.fixed_values().nu0();
  } else if (prior.has_ngw_prior()) {
    // TODO
  } else {
    std::invalid_argument("Error: argument proto is not appropriate");
  }
}

void NNWHierarchy::write_state_to_proto(google::protobuf::Message *out) const {
  bayesmix::MultiLSState state;
  to_proto(mean, state.mutable_mean());
  to_proto(tau, state.mutable_precision());

  google::protobuf::internal::down_cast<bayesmix::MarginalState::ClusterVal *>(
      out)
      ->mutable_multi_ls_state()
      ->CopyFrom(state);
}
