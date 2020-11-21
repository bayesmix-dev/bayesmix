#include "nnw_hierarchy.hpp"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/prob.hpp>

#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../../proto/cpp/matrix.pb.h"
#include "../utils/distributions.hpp"
#include "../utils/proto_utils.hpp"
#include "../utils/rng.hpp"

//! \param prec_ Value to set to prec
void NNWHierarchy::set_prec_and_utilities(const Eigen::MatrixXd &prec_) {
  state.prec = prec_;

  // Update prec utilities
  prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec_);
  prec_chol_eval = prec_chol.matrixL().transpose();
  Eigen::VectorXd diag = prec_chol_eval.diagonal();
  prec_logdet = 2 * log(diag.array()).sum();
}

void NNWHierarchy::check_spd(const Eigen::MatrixXd &mat) {
  assert(mat.rows() == mat.cols());
  assert(mat.isApprox(mat.transpose()) && "Error: matrix is not symmetric");
  stan::math::check_pos_definite("", "Matrix", mat);
}

void NNWHierarchy::initialize() {
  unsigned int dim = hypers->mu.size();
  state.mean = hypers->mu;
  set_prec_and_utilities(hypers->lambda * Eigen::MatrixXd::Identity(dim, dim));
}

//! \param data                    Matrix of row-vectorial data points
//! \param mu0, lambda0, tau0, nu0 Original values for hyperparameters
//! \return                        Vector of updated values for hyperparameters
NNWHierarchy::Hyperparams NNWHierarchy::normal_wishart_update(
    const Eigen::MatrixXd &data, const Eigen::VectorXd &mu0,
    const double lambda0, const Eigen::MatrixXd &tau0_inv, const double nu0) {
  // Initialize relevant objects
  Hyperparams post_params;
  unsigned int n = data.rows();

  // Compute updated hyperparameters
  post_params.lambda = lambda0 + n;
  post_params.nu = nu0 + 0.5 * n;

  Eigen::VectorXd mubar = data.colwise().mean();  // sample mean
  post_params.mu = (lambda0 * mu0 + n * mubar) / (lambda0 + n);
  // Compute tau_n
  Eigen::MatrixXd tau_temp = Eigen::MatrixXd::Zero(data.cols(), data.cols());
  for (size_t i = 0; i < n; i++) {
    Eigen::VectorXd datum = data.row(i);
    tau_temp += (datum - mubar) * (datum - mubar).transpose();  // column * row
  }
  tau_temp += (n * lambda0 / (n + lambda0)) * (mubar - mu0) *
              (mubar - mu0).transpose();
  tau_temp = 0.5 * tau_temp + tau0_inv;
  post_params.tau = stan::math::inverse_spd(tau_temp);
  return post_params;
}

void NNWHierarchy::update_hypers(
    const std::vector<bayesmix::MarginalState::ClusterVal> &states) {
  if (prior.has_fixed_values()) {
    return;
  } else if (prior.has_ngiw_prior()) {
    // Get hyperparameters:
    // for mu0
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior.ngiw_prior().mu0_prior().mu00());
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior.ngiw_prior().mu0_prior().sigma00());
    // for lambda0
    double alpha00 = prior.ngiw_prior().lambda0_prior().alpha00();
    double beta00 = prior.ngiw_prior().lambda0_prior().beta00();
    // for tau0
    double nu00 = prior.ngiw_prior().tau0_prior().nu00();
    Eigen::MatrixXd tau00 =
        bayesmix::to_eigen(prior.ngiw_prior().tau0_prior().tau00());
    // for nu0
    double nu0 = prior.ngiw_prior().nu0();

    // Compute posterior hyperparameters
    unsigned int dim = mu00.size();
    Eigen::MatrixXd sigma00inv = stan::math::inverse_spd(sigma00);
    Eigen::MatrixXd tau_n(dim, dim);
    Eigen::VectorXd num(dim);
    double beta_n = 0.0;
    for (auto &st : states) {
      Eigen::VectorXd mean = bayesmix::to_eigen(st.multi_ls_state().mean());
      Eigen::MatrixXd prec = bayesmix::to_eigen(st.multi_ls_state().prec());
      tau_n += prec;
      num += prec * mean;
      beta_n += (hypers->mu - mean).transpose() * prec * (hypers->mu - mean);
    }
    Eigen::MatrixXd prec = hypers->lambda * tau_n + sigma00inv;
    tau_n += tau00;
    num = hypers->lambda * num + sigma00inv * mu00;
    beta_n = beta00 + 0.5 * beta_n;
    Eigen::MatrixXd sig_n = stan::math::inverse_spd(prec);
    Eigen::VectorXd mu_n = sig_n * num;
    double alpha_n = alpha00 + 0.5 * states.size();
    double nu_n = nu00 + states.size() * hypers->nu;

    // Update hyperparameters with posterior random Gibbs sampling
    auto &rng = bayesmix::Rng::Instance().get();
    hypers->mu = stan::math::multi_normal_rng(mu_n, sig_n, rng);
    hypers->lambda = stan::math::gamma_rng(alpha_n, beta_n, rng);
    hypers->tau = stan::math::inv_wishart_rng(nu_n, tau_n, rng);
    tau0_inv = stan::math::inverse_spd(hypers->tau);
  } else {
    std::invalid_argument("Error: unrecognized prior");
  }
}

//! \param data Matrix of row-vectorial single data point
//! \return     Log-Likehood vector evaluated in data
double NNWHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
  // Initialize relevant objects
  return bayesmix::multi_normal_prec_lpdf(datum, state.mean, prec_chol_eval,
                                          prec_logdet);
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
    result(i) = bayesmix::multi_normal_prec_lpdf(data.row(i), state.mean,
                                                 prec_chol_eval, prec_logdet);
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
  state.mean = stan::math::multi_normal_prec_rng(
      hypers->mu, tau_new * hypers->lambda, rng);
  set_prec_and_utilities(tau_new);
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
  state.mean = stan::math::multi_normal_prec_rng(params.mu,
                                                 tau_new * params.lambda, rng);

  // Update state
  set_prec_and_utilities(tau_new);
}

void NNWHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  const bayesmix::MarginalState::ClusterVal &currcast =
      google::protobuf::internal::down_cast<
          const bayesmix::MarginalState::ClusterVal &>(state_);

  state.mean = to_eigen(currcast.multi_ls_state().mean());
  set_prec_and_utilities(to_eigen(currcast.multi_ls_state().prec()));
}

void NNWHierarchy::set_prior(const google::protobuf::Message &prior_) {
  const bayesmix::NNWPrior &currcast =
      google::protobuf::internal::down_cast<const bayesmix::NNWPrior &>(
          prior_);
  prior = currcast;
  hypers = std::make_shared<Hyperparams>();
  if (prior.has_fixed_values()) {
    // Set values
    hypers->mu = bayesmix::to_eigen(prior.fixed_values().mu0());
    hypers->lambda = prior.fixed_values().lambda0();
    hypers->tau = bayesmix::to_eigen(prior.fixed_values().tau0());
    tau0_inv = stan::math::inverse_spd(hypers->tau);
    hypers->nu = prior.fixed_values().nu0();
    // Check validity
    unsigned int dim = hypers->mu.size();
    assert(hypers->lambda > 0);
    assert(dim == hypers->tau.rows() &&
           "Error: hyperparameters dimensions are not consistent");
    assert(hypers->nu > dim - 1);
  } else if (prior.has_ngiw_prior()) {
    // Get hyperparameters:
    // for mu0
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior.ngiw_prior().mu0_prior().mu00());
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior.ngiw_prior().mu0_prior().sigma00());
    // for lambda0
    double alpha00 = prior.ngiw_prior().lambda0_prior().alpha00();
    double beta00 = prior.ngiw_prior().lambda0_prior().beta00();
    // for tau0
    double nu00 = prior.ngiw_prior().tau0_prior().nu00();
    Eigen::MatrixXd tau00 =
        bayesmix::to_eigen(prior.ngiw_prior().tau0_prior().tau00());
    // for nu0
    double nu0 = prior.ngiw_prior().nu0();

    // Check validity:
    // dimensionality
    unsigned int dim = mu00.size();
    assert(sigma00.rows() == dim &&
           "Error: hyperparameters dimensions are not consistent");
    assert(tau00.rows() == dim &&
           "Error: hyperparameters dimensions are not consistent");
    // for mu0
    check_spd(sigma00);
    // for lalmbda0
    assert(alpha00 > 0);
    assert(beta00 > 0);
    // for tau0
    assert(nu00 > 0);
    check_spd(tau00);
    // check nu0
    assert(nu0 > dim - 1);

    // Set initial values
    hypers->mu = mu00;
    hypers->lambda = alpha00 / beta00;
    hypers->tau = tau00 / (nu00 + dim + 1);
    tau0_inv = stan::math::inverse_spd(hypers->tau);
    hypers->nu = nu0;

  } else {
    std::invalid_argument("Error: unrecognized prior");
  }
}

void NNWHierarchy::write_state_to_proto(google::protobuf::Message *out) const {
  bayesmix::MultiLSState state_;
  to_proto(state.mean, state_.mutable_mean());
  to_proto(state.prec, state_.mutable_prec());

  google::protobuf::internal::down_cast<bayesmix::MarginalState::ClusterVal *>(
      out)
      ->mutable_multi_ls_state()
      ->CopyFrom(state_);
}
