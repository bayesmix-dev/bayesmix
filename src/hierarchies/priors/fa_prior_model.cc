#include "fa_prior_model.h"

double FAPriorModel::lpdf(const google::protobuf::Message &state_) {
  // Downcast state
  auto &state = downcast_state(state_).fa_state();

  // Proto2Eigen conversion
  Eigen::VectorXd mu = bayesmix::to_eigen(state.mu());
  Eigen::VectorXd psi = bayesmix::to_eigen(state.psi());

  // Eigen::MatrixXd eta = bayesmix::to_eigen(state.eta());
  Eigen::MatrixXd lambda = bayesmix::to_eigen(state.lambda());

  // Initialize lpdf value
  double target = 0.;

  // Compute lpdf
  for (size_t j = 0; j < dim; j++) {
    target +=
        stan::math::normal_lpdf(mu(j), hypers->mutilde(j), sqrt(hypers->phi));
    target +=
        stan::math::inv_gamma_lpdf(psi(j), hypers->alpha0, hypers->beta(j));
    for (size_t i = 0; i < hypers->q; i++) {
      target += stan::math::normal_lpdf(lambda(j, i), 0, 1);
    }
  }

  // Return lpdf contribution
  return target;
}

State::FA FAPriorModel::sample(ProtoHypersPtr hier_hypers) {
  // Random seed
  auto &rng = bayesmix::Rng::Instance().get();

  // Get params to use
  auto params = get_hypers_proto()->fa_state();
  Eigen::VectorXd mutilde = bayesmix::to_eigen(params.mutilde());
  Eigen::VectorXd beta = bayesmix::to_eigen(params.beta());

  // Compute output state
  State::FA out;
  out.mu = mutilde;
  out.psi = beta / (params.alpha0() + 1.);
  out.lambda = Eigen::MatrixXd::Zero(dim, params.q());
  for (size_t j = 0; j < dim; j++) {
    out.mu[j] = stan::math::normal_rng(mutilde[j], sqrt(params.phi()), rng);

    out.psi[j] = stan::math::inv_gamma_rng(params.alpha0(), beta[j], rng);

    for (size_t i = 0; i < params.q(); i++) {
      out.lambda(j, i) = stan::math::normal_rng(0, 1, rng);
    }
  }
  return out;
}

void FAPriorModel::update_hypers(
    const std::vector<bayesmix::AlgorithmState::ClusterState> &states) {
  auto &rng = bayesmix::Rng::Instance().get();
  if (prior->has_fixed_values()) {
    return;
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

void FAPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).fa_state();
  hypers->mutilde = bayesmix::to_eigen(hyperscast.mutilde());
  hypers->alpha0 = hyperscast.alpha0();
  hypers->beta = bayesmix::to_eigen(hyperscast.beta());
  hypers->phi = hyperscast.phi();
  hypers->q = hyperscast.q();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
FAPriorModel::get_hypers_proto() const {
  bayesmix::FAPriorDistribution hypers_;
  bayesmix::to_proto(hypers->mutilde, hypers_.mutable_mutilde());
  bayesmix::to_proto(hypers->beta, hypers_.mutable_beta());
  hypers_.set_alpha0(hypers->alpha0);
  hypers_.set_phi(hypers->phi);
  hypers_.set_q(hypers->q);

  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_fa_state()->CopyFrom(hypers_);
  return out;
}

void FAPriorModel::initialize_hypers() {
  if (prior->has_fixed_values()) {
    // Set values
    hypers->mutilde = bayesmix::to_eigen(prior->fixed_values().mutilde());
    dim = hypers->mutilde.size();
    hypers->beta = bayesmix::to_eigen(prior->fixed_values().beta());
    hypers->phi = prior->fixed_values().phi();
    hypers->alpha0 = prior->fixed_values().alpha0();
    hypers->q = prior->fixed_values().q();

    // Check validity
    if (dim != hypers->beta.rows()) {
      throw std::invalid_argument(
          "Hyperparameters dimensions are not consistent");
    }
    for (size_t j = 0; j < dim; j++) {
      if (hypers->beta[j] <= 0) {
        throw std::invalid_argument("Shape parameter must be > 0");
      }
    }
    if (hypers->alpha0 <= 0) {
      throw std::invalid_argument("Scale parameter must be > 0");
    }
    if (hypers->phi <= 0) {
      throw std::invalid_argument("Diffusion parameter must be > 0");
    }
    if (hypers->q <= 0) {
      throw std::invalid_argument("Number of factors must be > 0");
    }
  } else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}
