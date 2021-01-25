#include "lddp_uni_hierarchy.hpp"

#include <Eigen/Dense>
#include <stan/math/prim/err.hpp>

#include "../utils/eigen_utils.hpp"
#include "../utils/proto_utils.hpp"
#include "../utils/rng.hpp"

void LDDPUniHierarchy::initialize() {
  state.regression_coeffs = Eigen::VectorXd::Zero(dim);
  clear_data();
}

void LDDPUniHierarchy::clear_data() {
  mixed_prod = Eigen::VectorXd::Zero(dim);
  data_sum_squares = 0.0;
  covar_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
  card = 0;
  cluster_data_idx = std::set<int>();
}

void LDDPUniHierarchy::update_summary_statistics(
    const Eigen::VectorXd &datum, const Eigen::VectorXd &covariate, bool add) {
  if (add) {
    data_sum_squares += datum(0) * datum(0);
    covar_sum_squares += covariate.row(0).transpose() * covariate.row(0);
    mixed_prod += datum(0) * covariate.row(0);
  } else {
    data_sum_squares -= datum(0) * datum(0);
    covar_sum_squares -= covariate.row(0).transpose() * covariate.row(0);
    mixed_prod -= datum(0) * covariate.row(0);
  }
}

LDDPUniHierarchy::Hyperparams LDDPUniHierarchy::normal_invgamma_update() {
  Hyperparams post_params;

  post_params.var_scaling = covar_sum_squares + hypers->var_scaling;
  post_params.mean = stan::math::inverse_spd(post_params.var_scaling) *
                     (mixed_prod + hypers->var_scaling * hypers->mean);
  post_params.shape = hypers->shape + 0.5 * card;
  post_params.scale =
      hypers->scale +
      0.5 * (data_sum_squares +
             hypers->mean.transpose() * hypers->var_scaling * hypers->mean -
             post_params.mean.transpose() * post_params.var_scaling *
                 post_params.mean);
  return post_params;
}

void LDDPUniHierarchy::update_hypers(
    const std::vector<bayesmix::MarginalState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  }

  else {
    throw std::invalid_argument("Error: unrecognized prior");
  }
}

double LDDPUniHierarchy::like_lpdf(const Eigen::RowVectorXd &datum,
                                   const Eigen::RowVectorXd &covariate) const {
  return stan::math::normal_lpdf(
      datum(0), state.regression_coeffs.dot(covariate), sqrt(state.var));
}

Eigen::VectorXd LDDPUniHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = like_lpdf(data.row(i), covariates.row(i));
  }
  return result;
}

double LDDPUniHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum,
                                   const Eigen::RowVectorXd &covariate) const {
  double sig_n = sqrt(
      (1 + (covariate.transpose() *
            stan::math::inverse_spd(hypers->var_scaling) * covariate)(0)) *
      hypers->scale / hypers->shape);
  return stan::math::student_t_lpdf(datum(0), 2 * hypers->shape,
                                    covariate.dot(hypers->mean), sig_n);
}

Eigen::VectorXd LDDPUniHierarchy::marg_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = marg_lpdf(data.row(i), covariates.row(i));
  }
  return result;
}

void LDDPUniHierarchy::draw() {
  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(hypers->shape, hypers->scale, rng);
  state.regression_coeffs = stan::math::multi_normal_prec_rng(
      hypers->mean, hypers->var_scaling / state.var, rng);
}

void LDDPUniHierarchy::sample_given_data() {
  // Update values
  Hyperparams params = normal_invgamma_update();

  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(params.shape, params.scale, rng);
  state.regression_coeffs = stan::math::multi_normal_prec_rng(
      params.mean, params.var_scaling / state.var, rng);
}

void LDDPUniHierarchy::sample_given_data(const Eigen::MatrixXd &data,
                                         const Eigen::MatrixXd &covariates) {
  data_sum_squares = data.squaredNorm();
  covar_sum_squares = covariates.transpose() * covariates;
  mixed_prod = covariates.transpose() * data;
  card = data.rows();
  log_card = std::log(card);
  sample_given_data();
}

void LDDPUniHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::MarginalState::ClusterState &>(state_);
  state.regression_coeffs =
      bayesmix::to_eigen(statecast.univ_dep_ls_state().regression_coeffs());
  state.var = statecast.univ_dep_ls_state().var();
  set_card(statecast.cardinality());
}

void LDDPUniHierarchy::set_prior(const google::protobuf::Message &prior_) {
  auto &priorcast =
      google::protobuf::internal::down_cast<const bayesmix::LDDUniPrior &>(
          prior_);
  prior = std::make_shared<bayesmix::LDDUniPrior>(priorcast);
  hypers = std::make_shared<Hyperparams>();
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mean = bayesmix::to_eigen(prior->fixed_values().mean());
    dim = hypers->mean.size();
    hypers->var_scaling =
        bayesmix::to_eigen(prior->fixed_values().var_scaling());
    hypers->shape = prior->fixed_values().shape();
    hypers->scale = prior->fixed_values().scale();
    // Check validity
    bayesmix::check_spd(hypers->var_scaling);
    assert(hypers->shape > 0);
    assert(hypers->scale > 0);
  }

  else {
    throw std::invalid_argument("Error: unrecognized prior");
  }
}

void LDDPUniHierarchy::write_state_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::UnivDepLSState state_;
  bayesmix::to_proto(state.regression_coeffs,
                     state_.mutable_regression_coeffs());
  state_.set_var(state.var);

  auto *out_cast = google::protobuf::internal::down_cast<
      bayesmix::MarginalState::ClusterState *>(out);
  out_cast->mutable_univ_dep_ls_state()->CopyFrom(state_);
  out_cast->set_cardinality(card);
}

void LDDPUniHierarchy::write_hypers_to_proto(
    google::protobuf::Message *out) const {
  bayesmix::LDDUniPrior hypers_;
  bayesmix::to_proto(hypers->mean,
                     hypers_.mutable_fixed_values()->mutable_mean());
  bayesmix::to_proto(hypers->var_scaling,
                     hypers_.mutable_fixed_values()->mutable_var_scaling());
  hypers_.mutable_fixed_values()->set_shape(hypers->shape);
  hypers_.mutable_fixed_values()->set_scale(hypers->scale);

  google::protobuf::internal::down_cast<bayesmix::LDDUniPrior *>(out)
      ->mutable_fixed_values()
      ->CopyFrom(hypers_.fixed_values());
}
