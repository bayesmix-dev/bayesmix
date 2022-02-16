#include "nnw_updater.h"

#include "src/hierarchies/likelihoods/states.h"
#include "src/hierarchies/priors/hyperparams.h"

void NNWUpdater::compute_posterior_hypers(AbstractLikelihood& like,
                                          AbstractPriorModel& prior) {
  // Likelihood and Prior downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);
  
  // Getting required quantities from likelihood and prior
  int card = likecast.get_card();
  Eigen::VectorXd data_sum = likecast.get_data_sum();
  Eigen::MatrixXd data_sum_squares = likecast.get_data_sum_squares();
  auto hypers = priorcast.get_hypers();

  // No update possible
  if (card == 0) {
    priorcast.set_posterior_hypers(hypers);
    return;
  }

  // Compute posterior hyperparameters
  Hyperparams::NW post_params;
  post_params.var_scaling = hypers.var_scaling + card;
  post_params.deg_free = hypers.deg_free + card;
  Eigen::VectorXd mubar = data_sum.array() / card;  // sample mean
  post_params.mean = (hypers.var_scaling * hypers.mean + card * mubar) /
                     (hypers.var_scaling + card);
  // Compute tau_n
  Eigen::MatrixXd tau_temp =
      data_sum_squares - card * mubar * mubar.transpose();
  tau_temp += (card * hypers.var_scaling / (card + hypers.var_scaling)) *
              (mubar - hypers.mean) * (mubar - hypers.mean).transpose();
  post_params.scale_inv = tau_temp + hypers.scale_inv;
  post_params.scale = stan::math::inverse_spd(post_params.scale_inv);
  post_params.scale_chol = Eigen::LLT<Eigen::MatrixXd>(post_params.scale).matrixU();
  priorcast.set_posterior_hypers(post_params);
  return;
};
