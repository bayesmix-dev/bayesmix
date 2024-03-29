#include "nnxig_updater.h"

#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/hyperparams.h"

AbstractUpdater::ProtoHypersPtr NNxIGUpdater::compute_posterior_hypers(
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
    return priorcast.get_hypers_proto();
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
  return std::make_shared<ProtoHypers>(out);
}
