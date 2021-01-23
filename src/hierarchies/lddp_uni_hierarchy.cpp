#include "lddp_uni_hierarchy.hpp"

#include <stan/math/prim/prob.hpp>
#include "../utils/eigen_utils.hpp"
#include "../utils/rng.hpp"

void LDDPUniHierarchy::initialize() {
  state.mean = Eigen::VectorXd::Zero(hypers->mean.size());
}

void LDDPUniHierarchy::clear_data() {  // TODO
  data_sum = 0.0;
  data_sum_squares = 0.0;
  card = 0;
  cluster_data_idx.clear();
}

void LDDPUniHierarchy::update_summary_statistics(  // TODO
  const Eigen::VectorXd &datum, bool add) {
  if (add) {
    data_sum += datum(0);
    data_sum_squares += datum(0) * datum(0);
  } else {
    data_sum -= datum(0);
    data_sum_squares -= datum(0) * datum(0);
  }
}

LDDPUniHierarchy::Hyperparams LDDPUniHierarchy::some_update() {
  return LDDPUniHierarchy::Hyperparams();  // TODO
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

double LDDPUniHierarchy::like_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) const {
  return stan::math::normal_lpdf(datum(0), state.mean.dot(covariate),
    sqrt(state.var));
}

Eigen::VectorXd LDDPUniHierarchy::like_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  Eigen::VectorXd result(data.rows());
  for (size_t i = 0; i < data.rows(); i++) {
    result(i) = stan::math::normal_lpdf(data(i, 0), state.mean.dot(covariate),
      sqrt(state.var));
  }
  return result;
}



double LDDPUniHierarchy::marg_lpdf(
    const Eigen::RowVectorXd &datum,
    const Eigen::RowVectorXd &covariate) const {
  return Eigen::VectorXd(0,0);  // TODO
}
// double NNIGHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) const {
//   double sig_n = sqrt(hypers->scale * (hypers->var_scaling + 1) /
//                       (hypers->shape * hypers->var_scaling));
//   return stan::math::student_t_lpdf(datum(0), 2 * hypers->shape,
//                                     hypers->mean, sig_n);
// }
// double NNWHierarchy::marg_lpdf(const Eigen::RowVectorXd &datum) const {
//   double nu_n = 2 * hypers->deg_free - dim + 1;
//   Eigen::MatrixXd sigma_n = hypers->scale_inv *
//                             (hypers->deg_free - 0.5 * (dim - 1)) *
//                             hypers->var_scaling / (hypers->var_scaling + 1);
//   return stan::math::multi_student_t_lpdf(datum, nu_n, hypers->mean,
//                                           sigma_n);
// }

Eigen::VectorXd LDDPUniHierarchy::marg_lpdf_grid(  
    const Eigen::MatrixXd &data, const Eigen::MatrixXd &covariates) const {
  return Eigen::VectorXd(0,0);  // TODO
}



void LDDPUniHierarchy::draw() {
  // Generate new state values from their prior centering distribution
  auto &rng = bayesmix::Rng::Instance().get();
  state.var = stan::math::inv_gamma_rng(hypers->shape, hypers->scale, rng);
  state.mean = stan::math::multi_normal_prec_rng(
      hypers->mean, hypers->var_scaling / state.var, rng);
}

void LDDPUniHierarchy::sample_given_data() {
  return;  // TODO
}

void LDDPUniHierarchy::sample_given_data(const Eigen::MatrixXd &data) {
  return;  // TODO
}

void NNIGHierarchy::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = google::protobuf::internal::down_cast<
      const bayesmix::MarginalState::ClusterState &>(state_);
  state.mean = to_eigen(statecast.multi_ls_state().mean());
  state.var = statecast.univ_ls_state().var();
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
    hypers->var_scaling = bayesmix::to_eigen(
      prior->fixed_values().var_scaling());
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
  bayesmix::to_proto(state.mean, state_.mutable_mean());
  state_.set_var(state.var);

  auto *out_cast = google::protobuf::internal::down_cast<
      bayesmix::MarginalState::ClusterState *>(out);
  out_cast->mutable_multi_dep_ls_state()->CopyFrom(state_);
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
