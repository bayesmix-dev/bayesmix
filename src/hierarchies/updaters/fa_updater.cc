#include "fa_updater.h"

#include "src/utils/distributions.h"

void FAUpdater::draw(AbstractLikelihood& like, AbstractPriorModel& prior,
                     bool update_params) {
  // Likelihood and PriorModel downcast
  auto& likecast = static_cast<FALikelihood&>(like);
  auto& priorcast = static_cast<FAPriorModel&>(prior);
  // Sample from the full conditional of the fa hierarchy
  bool set_card = true, use_post_hypers = true;
  if (likecast.get_card() == 0) {
    likecast.set_state_from_proto(*priorcast.sample(), !set_card);
  } else {
    // Get state and hypers
    State::FA new_state = likecast.get_state();
    Hyperparams::FA hypers = priorcast.get_hypers();
    // Gibbs update
    sample_eta(new_state, hypers, likecast);
    sample_mu(new_state, hypers, likecast);
    sample_psi(new_state, hypers, likecast);
    sample_lambda(new_state, hypers, likecast);
    // Eigen2Proto conversion
    bayesmix::AlgorithmState::ClusterState new_state_proto;
    bayesmix::to_proto(new_state.eta,
                       new_state_proto.mutable_fa_state()->mutable_eta());
    bayesmix::to_proto(new_state.mu,
                       new_state_proto.mutable_fa_state()->mutable_mu());
    bayesmix::to_proto(new_state.psi,
                       new_state_proto.mutable_fa_state()->mutable_psi());
    bayesmix::to_proto(new_state.lambda,
                       new_state_proto.mutable_fa_state()->mutable_lambda());
    likecast.set_state_from_proto(new_state_proto, !set_card);
  }
}

void FAUpdater::sample_eta(State::FA& state, const Hyperparams::FA& hypers,
                           const FALikelihood& like) {
  // Random Seed
  auto& rng = bayesmix::Rng::Instance().get();
  // Get required information
  auto dataset_ptr = like.get_dataset();
  auto cluster_data_idx = like.get_data_idx();
  unsigned int card = like.get_card();
  // eta update
  state.eta = Eigen::MatrixXd::Zero(card, hypers.q);
  auto sigma_eta_inv_llt =
      (Eigen::MatrixXd::Identity(hypers.q, hypers.q) +
       state.lambda.transpose() * state.psi_inverse * state.lambda)
          .llt();
  Eigen::MatrixXd temp_product(
      sigma_eta_inv_llt.solve(state.lambda.transpose() * state.psi_inverse));
  auto iterator = cluster_data_idx.begin();
  for (size_t i = 0; i < card; i++, iterator++) {
    Eigen::VectorXd tempvector(dataset_ptr->row(
        *iterator));  // TODO use slicing when Eigen is updated to v3.4
    state.eta.row(i) = (bayesmix::multi_normal_prec_chol_rng(
        temp_product * (tempvector - state.mu), sigma_eta_inv_llt, rng));
  }
}

void FAUpdater::sample_mu(State::FA& state, const Hyperparams::FA& hypers,
                          const FALikelihood& like) {
  // Random seed
  auto& rng = bayesmix::Rng::Instance().get();
  // Get required information
  Eigen::VectorXd data_sum = like.get_data_sum();
  unsigned int card = like.get_card();
  // mu update
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> sigma_mu;
  sigma_mu.diagonal() =
      (card * state.psi_inverse.diagonal().array() + hypers.phi)
          .cwiseInverse();
  Eigen::VectorXd sum = (state.eta.colwise().sum());
  Eigen::VectorXd mumean =
      sigma_mu * (hypers.phi * hypers.mutilde +
                  state.psi_inverse * (data_sum - state.lambda * sum));
  state.mu = bayesmix::multi_normal_diag_rng(mumean, sigma_mu, rng);
}

void FAUpdater::sample_lambda(State::FA& state, const Hyperparams::FA& hypers,
                              const FALikelihood& like) {
  // Random seed
  auto& rng = bayesmix::Rng::Instance().get();
  // Getting required information
  unsigned int dim = like.get_dim();
  unsigned int card = like.get_card();
  auto dataset_ptr = like.get_dataset();
  auto cluster_data_idx = like.get_data_idx();
  // lambda update
  Eigen::MatrixXd temp_etateta(state.eta.transpose() * state.eta);
  for (size_t j = 0; j < dim; j++) {
    auto sigma_lambda_inv_llt =
        (Eigen::MatrixXd::Identity(hypers.q, hypers.q) +
         temp_etateta / state.psi[j])
            .llt();
    Eigen::VectorXd tempsum(card);
    const Eigen::VectorXd& data_col = dataset_ptr->col(j);
    auto iterator = cluster_data_idx.begin();
    for (size_t i = 0; i < card; i++, iterator++) {
      tempsum[i] = data_col(
          *iterator);  // TODO use slicing when Eigen is updated to v3.4
    }
    tempsum = tempsum.array() - state.mu[j];
    tempsum = tempsum.array() / state.psi[j];
    state.lambda.row(j) = bayesmix::multi_normal_prec_chol_rng(
        sigma_lambda_inv_llt.solve(state.eta.transpose() * tempsum),
        sigma_lambda_inv_llt, rng);
  }
}

void FAUpdater::sample_psi(State::FA& state, const Hyperparams::FA& hypers,
                           const FALikelihood& like) {
  // Random seed
  auto& rng = bayesmix::Rng::Instance().get();
  // Getting required information
  auto dataset_ptr = like.get_dataset();
  auto cluster_data_idx = like.get_data_idx();
  unsigned int dim = like.get_dim();
  unsigned int card = like.get_card();
  // psi update
  for (size_t j = 0; j < dim; j++) {
    double sum = 0;
    auto iterator = cluster_data_idx.begin();
    for (size_t i = 0; i < card; i++, iterator++) {
      sum += std::pow(
          ((*dataset_ptr)(*iterator, j) -
           state.mu[j] -  // TODO use slicing when Eigen is updated to v3.4
           state.lambda.row(j).dot(state.eta.row(i))),
          2);
    }
    state.psi[j] = stan::math::inv_gamma_rng(hypers.alpha0 + card / 2,
                                             hypers.beta[j] + sum / 2, rng);
  }
}
