#include "nnig_updater.h"

std::shared_ptr<NNIGUpdater> NNIGUpdater::clone() const {
  auto out =
      std::make_shared<NNIGUpdater>(static_cast<NNIGUpdater const &>(*this));
  return out;
};

void NNIGUpdater::initialize(UniNormLikelihood &like, NIGPriorModel &prior) {
  // PriorModel Initialization
  prior.initialize();
  Hyperparams::NIG hypers = prior.get_hypers();
  prior.set_posterior_hypers(hypers);

  // State initialization
  State::UniLS state;
  state.mean = hypers.mean;
  state.var = hypers.scale / (hypers.shape + 1);

  // Likelihood Initalization
  like.set_state(state);
  like.clear_data();
  like.clear_summary_statistics();
};

void NNIGUpdater::compute_posterior_hypers(UniNormLikelihood &like,
                                           NIGPriorModel &prior) {
  // std::cout << "NNIGUpdater::compute_posterior_hypers()" << std::endl;
  // Getting required quantities from likelihood and prior
  int card = like.get_card();
  double data_sum = like.get_data_sum();
  double data_sum_squares = like.get_data_sum_squares();
  auto hypers = prior.get_hypers();

  // std::cout << "current cardinality: " << card << std::endl;

  // No update possible
  if (card == 0) {
    prior.set_posterior_hypers(hypers);
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

  prior.set_posterior_hypers(post_params);
  return;
};

void NNIGUpdater::draw(UniNormLikelihood &like, NIGPriorModel &prior,
                       bool update_params) {
  if (like.get_card() == 0) {
    like.set_state_from_proto(*prior.sample(false));
  } else {
    if (update_params) {
      compute_posterior_hypers(like, prior);
    }
    like.set_state_from_proto(*prior.sample(true));
  }
};
