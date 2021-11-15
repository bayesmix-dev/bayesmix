#include "nnw_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "matrix.pb.h"
#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

double NNWHierarchy::like_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  return bayesmix::multi_normal_prec_lpdf(datum, state.mean, state.prec_chol,
                                          state.prec_logdet);
}

double NNWHierarchy::marg_lpdf(
    const NNW::Hyperparams &params, const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
  NNW::Hyperparams pred_params = get_predictive_t_parameters(params);
  Eigen::VectorXd diag = pred_params.scale_chol.diagonal();
  double logdet = 2 * log(diag.array()).sum();

  return bayesmix::multi_student_t_invscale_lpdf(
      datum, pred_params.deg_free, pred_params.mean, pred_params.scale_chol,
      logdet);
}

Eigen::VectorXd NNWHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  // Custom, optimized grid method
  return bayesmix::multi_normal_prec_lpdf_grid(
      data, state.mean, state.prec_chol, state.prec_logdet);
}

Eigen::VectorXd NNWHierarchy::prior_pred_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  // Custom, optimized grid method
  NNW::Hyperparams pred_params = get_predictive_t_parameters(*hypers);
  Eigen::VectorXd diag = pred_params.scale_chol.diagonal();
  double logdet = 2 * log(diag.array()).sum();

  return bayesmix::multi_student_t_invscale_lpdf_grid(
      data, pred_params.deg_free, pred_params.mean, pred_params.scale_chol,
      logdet);
}

Eigen::VectorXd NNWHierarchy::conditional_pred_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  // Custom, optimized grid method
  NNW::Hyperparams pred_params =
      get_predictive_t_parameters(get_posterior_parameters());
  Eigen::VectorXd diag = pred_params.scale_chol.diagonal();
  double logdet = 2 * log(diag.array()).sum();

  return bayesmix::multi_student_t_invscale_lpdf_grid(
      data, pred_params.deg_free, pred_params.mean, pred_params.scale_chol,
      logdet);
}

void NNWHierarchy::initialize_state() {
  state.mean = hypers->mean;
  write_prec_to_state(
      hypers->var_scaling * Eigen::MatrixXd::Identity(dim, dim), &state);
}

void NNWHierarchy::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = bayesmix::to_eigen(prior->fixed_values().mean());
    dim = hypers->mean.size();
    hypers->var_scaling = prior->fixed_values().var_scaling();
    hypers->scale = bayesmix::to_eigen(prior->fixed_values().scale());
    hypers->deg_free = prior->fixed_values().deg_free();
    // Check validity
    if (hypers->var_scaling <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    if (dim != hypers->scale.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    if (hypers->deg_free <= dim - 1) {
      throw std::invalid_argument("Degrees of freedom parameter is not valid");
    }
  }

  else if (prior->has_normal_mean_prior()) {
    // Get hyperparameters
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().mean());
    dim = mu00.size();
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().var());
    double lambda0 = prior->normal_mean_prior().var_scaling();
    Eigen::MatrixXd tau0 =
        bayesmix::to_eigen(prior->normal_mean_prior().scale());
    double nu0 = prior->normal_mean_prior().deg_free();
    // Check validity
    unsigned int dim = mu00.size();
    if (sigma00.rows() != dim or tau0.rows() != dim) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    bayesmix::check_spd(sigma00);
    if (lambda0 <= 0) {
      throw std::invalid_argument("Variance-scaling parameter must be > 0");
    }
    bayesmix::check_spd(tau0);
    if (nu0 <= dim - 1) {
      throw std::invalid_argument("Degrees of freedom parameter is not valid");
    }
    // Set initial values
    hypers->mean = mu00;
    hypers->var_scaling = lambda0;
    hypers->scale = tau0;
    hypers->deg_free = nu0;
  }

  else if (prior->has_ngiw_prior()) {
    // Get hyperparameters:
    // for mu0
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().mean());
    dim = mu00.size();
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().var());
    // for lambda0
    double alpha00 = prior->ngiw_prior().var_scaling_prior().shape();
    double beta00 = prior->ngiw_prior().var_scaling_prior().rate();
    // for tau0
    double nu00 = prior->ngiw_prior().scale_prior().deg_free();
    Eigen::MatrixXd tau00 =
        bayesmix::to_eigen(prior->ngiw_prior().scale_prior().scale());
    // for nu0
    double nu0 = prior->ngiw_prior().deg_free();
    // Check validity:
    // dimensionality
    if (sigma00.rows() != dim or tau00.rows() != dim) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    // for mu0
    bayesmix::check_spd(sigma00);
    // for lambda0
    if (alpha00 <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (beta00 <= 0) {
      throw std::invalid_argument("Rate parameter must be > 0");
    }
    // for tau0
    if (nu00 <= 0) {
      throw std::invalid_argument("Degrees of freedom parameter must be > 0");
    }
    bayesmix::check_spd(tau00);
    // check nu0
    if (nu0 <= dim - 1) {
      throw std::invalid_argument("Degrees of freedom parameter is not valid");
    }
    // Set initial values
    hypers->mean = mu00;
    hypers->var_scaling = alpha00 / beta00;
    hypers->scale = tau00 / (nu00 + dim + 1);
    hypers->deg_free = nu0;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
  hypers->scale_inv = stan::math::inverse_spd(hypers->scale);
  hypers->scale_chol = Eigen::LLT<Eigen::MatrixXd>(hypers->scale).matrixU();
}

void NNWHierarchy::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else if (prior->has_normal_mean_prior()) {
    // Get hyperparameters
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().mean());
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->normal_mean_prior().mean_prior().var());
    double lambda0 = prior->normal_mean_prior().var_scaling();
    // Compute posterior hyperparameters
    Eigen::MatrixXd sigma00inv = stan::math::inverse_spd(sigma00);
    Eigen::MatrixXd prec = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd num = Eigen::MatrixXd::Zero(dim, 1);
    for (auto &st : states) {
      Eigen::MatrixXd prec_i = bayesmix::to_eigen(st.multi_ls_state().prec());
      prec += prec_i;
      num += prec_i * bayesmix::to_eigen(st.multi_ls_state().mean());
    }
    prec = hypers->var_scaling * prec + sigma00inv;
    num = hypers->var_scaling * num + sigma00inv * mu00;
    Eigen::VectorXd mu_n = prec.llt().solve(num);
    // Update hyperparameters with posterior sampling
    hypers->mean = stan::math::multi_normal_prec_rng(mu_n, prec, rng);
  }

  else if (prior->has_ngiw_prior()) {
    // Get hyperparameters:
    // for mu0
    Eigen::VectorXd mu00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().mean());
    Eigen::MatrixXd sigma00 =
        bayesmix::to_eigen(prior->ngiw_prior().mean_prior().var());
    // for lambda0
    double alpha00 = prior->ngiw_prior().var_scaling_prior().shape();
    double beta00 = prior->ngiw_prior().var_scaling_prior().rate();
    // for tau0
    double nu00 = prior->ngiw_prior().scale_prior().deg_free();
    Eigen::MatrixXd tau00 =
        bayesmix::to_eigen(prior->ngiw_prior().scale_prior().scale());
    // Compute posterior hyperparameters
    Eigen::MatrixXd sigma00inv = stan::math::inverse_spd(sigma00);
    Eigen::MatrixXd tau_n = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd num = Eigen::MatrixXd::Zero(dim, 1);
    double beta_n = 0.0;
    for (auto &st : states) {
      Eigen::VectorXd mean = bayesmix::to_eigen(st.multi_ls_state().mean());
      Eigen::MatrixXd prec = bayesmix::to_eigen(st.multi_ls_state().prec());
      tau_n += prec;
      num += prec * mean;
      beta_n +=
          (hypers->mean - mean).transpose() * prec * (hypers->mean - mean);
    }
    Eigen::MatrixXd prec_n = hypers->var_scaling * tau_n + sigma00inv;
    tau_n += tau00;
    num = hypers->var_scaling * num + sigma00inv * mu00;
    beta_n = beta00 + 0.5 * beta_n;
    Eigen::MatrixXd sig_n = stan::math::inverse_spd(prec_n);
    Eigen::VectorXd mu_n = sig_n * num;
    double alpha_n = alpha00 + 0.5 * states.size();
    double nu_n = nu00 + states.size() * hypers->deg_free;
    // Update hyperparameters with posterior random Gibbs sampling
    hypers->mean = stan::math::multi_normal_rng(mu_n, sig_n, rng);
    hypers->var_scaling = stan::math::gamma_rng(alpha_n, beta_n, rng);
    hypers->scale = stan::math::inv_wishart_rng(nu_n, tau_n, rng);
    hypers->scale_inv = stan::math::inverse_spd(hypers->scale);
    hypers->scale_chol = Eigen::LLT<Eigen::MatrixXd>(hypers->scale).matrixU();
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

NNW::State NNWHierarchy::draw(const NNW::Hyperparams &params) {
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(params.deg_free, params.scale, rng);
  // Update state
  NNW::State out;
  out.mean = stan::math::multi_normal_prec_rng(
      params.mean, tau_new * params.var_scaling, rng);
  write_prec_to_state(tau_new, &out);
  return out;
}

void NNWHierarchy::update_summary_statistics(
    const Eigen::RowVectorXd &datum, const Eigen::RowVectorXd &covariate,
    bool add) {
  if (add) {
    data_sum += datum.transpose();
    data_sum_squares += datum.transpose() * datum;
  } else {
    data_sum -= datum.transpose();
    data_sum_squares -= datum.transpose() * datum;
  }
}

void NNWHierarchy::clear_summary_statistics() {
  data_sum = Eigen::VectorXd::Zero(dim);
  data_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}

NNW::Hyperparams NNWHierarchy::get_posterior_parameters() const {
  if (card == 0) {  // no update possible
    return *hypers;
  }
  // Compute posterior hyperparameters
  NNW::Hyperparams post_params;
  post_params.var_scaling = hypers->var_scaling + card;
  post_params.deg_free = hypers->deg_free + card;
  Eigen::VectorXd mubar = data_sum.array() / card;  // sample mean
  post_params.mean = (hypers->var_scaling * hypers->mean + card * mubar) /
                     (hypers->var_scaling + card);
  // Compute tau_n
  Eigen::MatrixXd tau_temp =
      data_sum_squares - card * mubar * mubar.transpose();
  tau_temp += (card * hypers->var_scaling / (card + hypers->var_scaling)) *
              (mubar - hypers->mean) * (mubar - hypers->mean).transpose();
  post_params.scale_inv = tau_temp + hypers->scale_inv;
  post_params.scale = stan::math::inverse_spd(post_params.scale_inv);
  post_params.scale_chol =
      Eigen::LLT<Eigen::MatrixXd>(post_params.scale).matrixU();
  return post_params;
}

void NNWHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::AlgorithmState::ClusterState &>(state_);
  state.mean = to_eigen(statecast.multi_ls_state().mean());
  state.prec = to_eigen(statecast.multi_ls_state().prec());
  state.prec_chol = to_eigen(statecast.multi_ls_state().prec());
  Eigen::VectorXd diag = state.prec_chol.diagonal();
  state.prec_logdet = 2 * log(diag.array()).sum();
  set_card(statecast.cardinality());
}

std::shared_ptr<google::protobuf::Message> NNWHierarchy::get_state_proto()
    const {
  auto out = std::make_unique<bayesmix::MultiLSState>();
  bayesmix::to_proto(state.mean, out->mutable_mean());
  bayesmix::to_proto(state.prec, out->mutable_prec());
  bayesmix::to_proto(state.prec_chol, out->mutable_prec_chol());
  return out;
}

void NNWHierarchy::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::NNWPrior hypers_;
  bayesmix::to_proto(hypers->mean,
                     hypers_.mutable_fixed_values()->mutable_mean());
  hypers_.mutable_fixed_values()->set_var_scaling(hypers->var_scaling);
  hypers_.mutable_fixed_values()->set_deg_free(hypers->deg_free);
  bayesmix::to_proto(hypers->scale,
                     hypers_.mutable_fixed_values()->mutable_scale());

  google::protobuf::internal::down_cast<bayesmix::NNWPrior *>(out)
      ->mutable_fixed_values()
      ->CopyFrom(hypers_.fixed_values());
}

void NNWHierarchy::write_prec_to_state(const Eigen::MatrixXd &prec_,
                                       NNW::State *out) {
  out->prec = prec_;
  // Update prec utilities
  out->prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec_).matrixU();
  Eigen::VectorXd diag = out->prec_chol.diagonal();
  out->prec_logdet = 2 * log(diag.array()).sum();
}

NNW::Hyperparams NNWHierarchy::get_predictive_t_parameters(
    const NNW::Hyperparams &params) const {
  // Compute dof and scale of marginal distribution
  double nu_n = params.deg_free - dim + 1;
  double coeff = (params.var_scaling + 1) / (params.var_scaling * nu_n);
  Eigen::MatrixXd scale_chol_n = params.scale_chol / std::sqrt(coeff);

  NNW::Hyperparams out;
  out.mean = params.mean;
  out.deg_free = nu_n;
  out.scale_chol = scale_chol_n;
  return out;
}
