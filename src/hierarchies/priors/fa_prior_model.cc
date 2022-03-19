#include "fa_prior_model.h"

double FAPriorModel::lpdf(const google::protobuf::Message &state_) {
  // Downcast state
  auto &state = downcast_state(state_).fa_state();
  // Proto to Eigen conversion
  Eigen::VectorXd mu = bayesmix::to_eigen(state.mu());
  Eigen::VectorXd psi = bayesmix::to_eigen(state.psi());
  Eigen::MatrixXd eta = bayesmix::to_eigen(state.eta());
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
  for (size_t i = 0; i < eta.rows(); i++) {
    for (size_t j = 0; j < hypers->q; j++) {
      target += stan::math::normal_lpdf(eta(i, j), 0, 1);
    }
  }
  // Return lpdf contribution
  return target;
}

std::shared_ptr<google::protobuf::Message> FAPriorModel::sample(
    bool use_post_hypers) {
  // Random seed
  auto &rng = bayesmix::Rng::Instance().get();

  // Select params to use
  Hyperparams::FA params = use_post_hypers ? post_hypers : *hypers;

  // HO AGGIUNTO PARAMS.CARD MA NON SO SE SIA LA SCELTA MIGLIORE!!!
  // Compute output state
  State::FA out;
  out.mu = params.mutilde;
  out.psi = params.beta / (params.alpha0 + 1.);
  out.eta = Eigen::MatrixXd::Zero(params.card, params.q);
  out.lambda = Eigen::MatrixXd::Zero(dim, params.q);
  for (size_t j = 0; j < dim; j++) {
    out.mu[j] =
        stan::math::normal_rng(params.mutilde[j], sqrt(params.phi), rng);

    out.psi[j] = stan::math::inv_gamma_rng(params.alpha0, params.beta[j], rng);

    for (size_t i = 0; i < params.q; i++) {
      out.lambda(j, i) = stan::math::normal_rng(0, 1, rng);
    }
  }
  for (size_t i = 0; i < params.card; i++) {
    for (size_t j = 0; j < params.q; j++) {
      out.eta(i, j) = stan::math::normal_rng(0, 1, rng);
    }
  }

  // Questi conti non li passo al proto, attenzione !!!
  // out.psi_inverse = out.psi.cwiseInverse().asDiagonal();
  // compute_wood_factors(out.cov_wood, out.cov_logdet, out.lambda,
  //                      out.psi_inverse);

  // Convert to proto
  bayesmix::AlgorithmState::ClusterState state;
  bayesmix::to_proto(out.mu, state.mutable_fa_state()->mutable_mu());
  bayesmix::to_proto(out.psi, state.mutable_fa_state()->mutable_psi());
  bayesmix::to_proto(out.eta, state.mutable_fa_state()->mutable_eta());
  bayesmix::to_proto(out.lambda, state.mutable_fa_state()->mutable_lambda());
  return std::make_shared<bayesmix::AlgorithmState::ClusterState>(state);

  // MANCA PSI_INVERSE E GLI OUTPUT DA COMPUTE_WOOD_FACTORS !!! BISOGNA
  // CAMBIARE IL PROTO
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
  hypers->card = hyperscast.card();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
FAPriorModel::get_hypers_proto() const {
  bayesmix::FAPriorDistribution hypers_;
  bayesmix::to_proto(hypers->mutilde, hypers_.mutable_mutilde());
  bayesmix::to_proto(hypers->beta, hypers_.mutable_beta());
  hypers_.set_alpha0(hypers->alpha0);
  hypers_.set_phi(hypers->phi);
  hypers_.set_q(hypers->q);
  hypers_.set_card(hypers->card);

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
    hypers->card = prior->fixed_values().card();

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
    if (hypers->card <= 0) {
      throw std::invalid_argument("Number of data must be > 0");
    }
  }

  else {
    throw std::invalid_argument("Unrecognized hierarchy prior");
  }
}

/*
// Automatic initialization
if (dim == 0) {
  hypers->mutilde = dataset_ptr->colwise().mean();
  dim = hypers->mutilde.size();
}
if (hypers->beta.size() == 0) {
  Eigen::MatrixXd centered =
      dataset_ptr->rowwise() - dataset_ptr->colwise().mean();
  auto cov_llt = ((centered.transpose() * centered) /
                  double(dataset_ptr->rows() - 1.))
                      .llt();
  Eigen::MatrixXd precision_matrix(
      cov_llt.solve(Eigen::MatrixXd::Identity(dim, dim)));
  hypers->beta =
      (hypers->alpha0 - 1) * precision_matrix.diagonal().cwiseInverse();
  if (hypers->alpha0 == 1) {
    throw std::invalid_argument(
        "Scale parameter must be different than 1 when automatic "
        "initialization is used");
  }
}
*/
