#include "nnig_updater.h"

#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/hyperparams.h"

void NNIGUpdater::compute_posterior_hypers(AbstractLikelihood& like,
                                           AbstractPriorModel& prior) {
  // Likelihood and Prior downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);

  // Getting required quantities from likelihood and prior
  int card = likecast.get_card();
  double data_sum = likecast.get_data_sum();
  double data_sum_squares = likecast.get_data_sum_squares();
  auto hypers = priorcast.get_hypers();

  // No update possible
  if (card == 0) {
    priorcast.set_posterior_hypers(hypers);
    return;
  }

  // Compute posterior hyperparameters
  Hyperparams::NIG post_params;
  double y_bar = data_sum / (1.0 * card);  // sample mean
  double ss = data_sum_squares - card * y_bar * y_bar;
  post_params.mean = (hypers.var_scaling * hypers.mean + data_sum) /
                     (hypers.var_scaling + card);
  post_params.var_scaling = hypers.var_scaling + card;
  post_params.shape = hypers.shape + 0.5 * card;
  post_params.scale = hypers.scale + 0.5 * ss +
                      0.5 * hypers.var_scaling * card * (y_bar - hypers.mean) *
                          (y_bar - hypers.mean) / (card + hypers.var_scaling);

  priorcast.set_posterior_hypers(post_params);
  return;
};
