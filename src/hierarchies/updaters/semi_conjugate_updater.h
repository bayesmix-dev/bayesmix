#ifndef BAYESMIX_HIERARCHIES_UPDATERS_SEMI_CONJUGATE_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_SEMI_CONJUGATE_UPDATER_H_

#include <utility>

#include "abstract_updater.h"
#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

template <class Likelihood, class PriorModel>
class SemiConjugateUpdater : public AbstractUpdater {
 public:
  SemiConjugateUpdater() = default;

  ~SemiConjugateUpdater() = default;

  void draw(AbstractLikelihood& like, AbstractPriorModel& prior,
            bool update_params) override;

  void save_posterior_hypers(const ProtoHypers& post_hypers_) override;

 protected:
  Likelihood& downcast_likelihood(AbstractLikelihood& like_);
  PriorModel& downcast_prior(AbstractPriorModel& prior_);
  ProtoHypers post_hypers;
};

// Methods' definitions
template <class Likelihood, class PriorModel>
Likelihood& SemiConjugateUpdater<Likelihood, PriorModel>::downcast_likelihood(
    AbstractLikelihood& like_) {
  return static_cast<Likelihood&>(like_);
}

template <class Likelihood, class PriorModel>
PriorModel& SemiConjugateUpdater<Likelihood, PriorModel>::downcast_prior(
    AbstractPriorModel& prior_) {
  return static_cast<PriorModel&>(prior_);
}

template <class Likelihood, class PriorModel>
void SemiConjugateUpdater<Likelihood, PriorModel>::draw(
    AbstractLikelihood& like, AbstractPriorModel& prior, bool update_params) {
  // Likelihood and PriorModel downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);
  // Sample from the full conditional of a semi-conjugate hierarchy
  bool set_card = true; /*, use_post_hypers=true;*/
  if (likecast.get_card() == 0) {
    auto prior_params = *priorcast.get_hypers_proto();
    likecast.set_state_from_proto(*priorcast.sample(prior_params), !set_card);
  } else {
    auto post_params = compute_posterior_hypers(likecast, priorcast);
    likecast.set_state_from_proto(*priorcast.sample(post_params), !set_card);
    if (update_params) save_posterior_hypers(post_params);
  }
}

template <class Likelihood, class PriorModel>
void SemiConjugateUpdater<Likelihood, PriorModel>::save_posterior_hypers(
    const ProtoHypers& post_hypers_) {
  post_hypers = post_hypers_;
  return;
}

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_SEMI_CONJUGATE_UPDATER_H_
