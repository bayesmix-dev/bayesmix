#include "mnig_updater.h"

void MNIGUpdater::compute_posterior_hypers(AbstractLikelihood& like,
                                           AbstractPriorModel& prior) {
  // Likelihood and Prior downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);

  // Getting required quantities from likelihood and prior
  int card = likecast.get_card();
  unsigned int dim = likecast.get_dim();
  double data_sum_squares = likecast.get_data_sum_squares();
  Eigen::MatrixXd covar_sum_squares = likecast.get_covar_sum_squares();
  Eigen::MatrixXd mixed_prod = likecast.get_mixed_prod();
  auto hypers = priorcast.get_hypers();

  // No update possible
  if (card == 0) {
    priorcast.set_posterior_hypers(hypers);
    return;
  }

  // Compute posterior hyperparameters
  Hyperparams::MNIG post_params;
  post_params.var_scaling = covar_sum_squares + hypers.var_scaling;
  auto llt = post_params.var_scaling.llt();
  post_params.var_scaling_inv = llt.solve(Eigen::MatrixXd::Identity(dim, dim));
  post_params.mean = llt.solve(mixed_prod + hypers.var_scaling * hypers.mean);
  post_params.shape = hypers.shape + 0.5 * card;
  post_params.scale =
      hypers.scale +
      0.5 * (data_sum_squares +
             hypers.mean.transpose() * hypers.var_scaling * hypers.mean -
             post_params.mean.transpose() * post_params.var_scaling *
                 post_params.mean);

  priorcast.set_posterior_hypers(post_params);
  return;
};
