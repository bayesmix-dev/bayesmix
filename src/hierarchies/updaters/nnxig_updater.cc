#include "nnxig_updater.h"

#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/hyperparams.h"

AbstractUpdater::ProtoHypers NNxIGUpdater::compute_posterior_hypers(
    AbstractLikelihood& like, AbstractPriorModel& prior) {
  // Likelihood and Prior downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);

  // Getting required quantities from likelihood and prior
  auto state = likecast.get_state();
  int card = likecast.get_card();
  double data_sum = likecast.get_data_sum();
  double data_sum_squares = likecast.get_data_sum_squares();
  auto hypers = priorcast.get_hypers();

  // No update possible
  if (card == 0) {
    return *(priorcast.get_hypers_proto());
  }

  // Compute posterior hyperparameters
  double mean, var, shape, scale;
  double var_y = data_sum_squares - 2 * state.mean * data_sum +
                 card * state.mean * state.mean;
  mean = (hypers.var * data_sum + state.var * hypers.mean) /
         (card * hypers.var + state.var);
  var = (state.var * hypers.var) / (card * hypers.var + state.var);
  shape = hypers.shape + 0.5 * card;
  scale = hypers.scale + 0.5 * var_y;

  // Proto conversion
  ProtoHypers out;
  out.mutable_nnxig_state()->set_mean(mean);
  out.mutable_nnxig_state()->set_var(var);
  out.mutable_nnxig_state()->set_shape(shape);
  out.mutable_nnxig_state()->set_scale(scale);
  return out;
}

// void NNxIGUpdater::compute_posterior_hypers(AbstractLikelihood& like,
//                                             AbstractPriorModel& prior) {
//   // Likelihood and Prior downcast
//   auto& likecast = downcast_likelihood(like);
//   auto& priorcast = downcast_prior(prior);

//   // Getting required quantities from likelihood and prior
//   auto state = likecast.get_state();
//   int card = likecast.get_card();
//   double data_sum = likecast.get_data_sum();
//   double data_sum_squares = likecast.get_data_sum_squares();
//   auto hypers = priorcast.get_hypers();

//   // No update possible
//   if (card == 0) {
//     priorcast.set_posterior_hypers(hypers);
//   }

//   // Compute posterior hyperparameters
//   Hyperparams::NxIG post_params;
//   double var_y = data_sum_squares - 2 * state.mean * data_sum +
//                  card * state.mean * state.mean;
//   post_params.mean = (hypers.var * data_sum + state.var * hypers.mean) /
//                      (card * hypers.var + state.var);
//   post_params.var = (state.var * hypers.var) / (card * hypers.var +
//   state.var); post_params.shape = hypers.shape + 0.5 * card;
//   post_params.scale = hypers.scale + 0.5 * var_y;
//   priorcast.set_posterior_hypers(post_params);
//   return;
// };
