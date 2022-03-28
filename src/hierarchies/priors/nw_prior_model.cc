#include "nw_prior_model.h"

#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

void NWPriorModel::initialize_hypers() {
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
    dim = mu00.size();
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
    unsigned int dim = mu00.size();
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

double NWPriorModel::lpdf(const google::protobuf::Message &state_) {
  auto &state = downcast_state(state_).multi_ls_state();
  Eigen::VectorXd mean = bayesmix::to_eigen(state.mean());
  Eigen::MatrixXd prec = bayesmix::to_eigen(state.prec());
  double target =
      stan::math::multi_normal_prec_lpdf(mean, hypers->mean,
                                         prec * hypers->var_scaling) +
      stan::math::wishart_lpdf(prec, hypers->deg_free, hypers->scale);
  return target;
}

std::shared_ptr<google::protobuf::Message> NWPriorModel::sample(
    ProtoHypersPtr hier_hypers) {
  auto &rng = bayesmix::Rng::Instance().get();
  auto params = (hier_hypers) ? hier_hypers->nnw_state()
                              : get_hypers_proto()->nnw_state();
  Eigen::MatrixXd scale = bayesmix::to_eigen(params.scale());
  Eigen::VectorXd mean = bayesmix::to_eigen(params.mean());

  Eigen::MatrixXd tau_new =
      stan::math::wishart_rng(params.deg_free(), scale, rng);

  // Update state
  State::MultiLS out;
  out.mean = stan::math::multi_normal_prec_rng(
      mean, tau_new * params.var_scaling(), rng);
  write_prec_to_state(tau_new, &out);

  // Make output state
  bayesmix::AlgorithmState::ClusterState state;
  bayesmix::to_proto(out.mean, state.mutable_multi_ls_state()->mutable_mean());
  bayesmix::to_proto(out.prec, state.mutable_multi_ls_state()->mutable_prec());
  bayesmix::to_proto(out.prec_chol,
                     state.mutable_multi_ls_state()->mutable_prec_chol());
  return std::make_shared<bayesmix::AlgorithmState::ClusterState>(state);
};

// std::shared_ptr<google::protobuf::Message> NWPriorModel::sample(
//     bool use_post_hypers) {
//   auto &rng = bayesmix::Rng::Instance().get();

//   Hyperparams::NW params = use_post_hypers ? post_hypers : *hypers;

//   Eigen::MatrixXd tau_new =
//       stan::math::wishart_rng(params.deg_free, params.scale, rng);

//   // Update state
//   State::MultiLS out;
//   out.mean = stan::math::multi_normal_prec_rng(
//       params.mean, tau_new * params.var_scaling, rng);
//   write_prec_to_state(tau_new, &out);

//   // Make output state
//   bayesmix::AlgorithmState::ClusterState state;
//   bayesmix::to_proto(out.mean,
//   state.mutable_multi_ls_state()->mutable_mean());
//   bayesmix::to_proto(out.prec,
//   state.mutable_multi_ls_state()->mutable_prec());
//   bayesmix::to_proto(out.prec_chol,
//                      state.mutable_multi_ls_state()->mutable_prec_chol());
//   return std::make_shared<bayesmix::AlgorithmState::ClusterState>(state);
// };

void NWPriorModel::update_hypers(
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

void NWPriorModel::set_hypers_from_proto(
    const google::protobuf::Message &hypers_) {
  auto &hyperscast = downcast_hypers(hypers_).nnw_state();
  hypers->mean = bayesmix::to_eigen(hyperscast.mean());
  hypers->var_scaling = hyperscast.var_scaling();
  hypers->deg_free = hyperscast.deg_free();
  hypers->scale = bayesmix::to_eigen(hyperscast.scale());
  hypers->scale_inv = stan::math::inverse_spd(hypers->scale);
  hypers->scale_chol = Eigen::LLT<Eigen::MatrixXd>(hypers->scale).matrixU();
}

std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>
NWPriorModel::get_hypers_proto() const {
  // Translate to proto
  bayesmix::Vector mean_proto;
  bayesmix::Matrix scale_proto;
  bayesmix::to_proto(hypers->mean, &mean_proto);
  bayesmix::to_proto(hypers->scale, &scale_proto);

  // Make output state and return
  auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
  out->mutable_nnw_state()->mutable_mean()->CopyFrom(mean_proto);
  out->mutable_nnw_state()->set_var_scaling(hypers->var_scaling);
  out->mutable_nnw_state()->set_deg_free(hypers->deg_free);
  out->mutable_nnw_state()->mutable_scale()->CopyFrom(scale_proto);
  return out;
}

void NWPriorModel::write_prec_to_state(const Eigen::MatrixXd &prec_,
                                       State::MultiLS *out) {
  out->prec = prec_;
  // Update prec utilities
  out->prec_chol = Eigen::LLT<Eigen::MatrixXd>(prec_).matrixU();
  Eigen::VectorXd diag = out->prec_chol.diagonal();
  out->prec_logdet = 2 * log(diag.array()).sum();
}
