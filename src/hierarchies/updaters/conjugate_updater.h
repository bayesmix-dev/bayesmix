#ifndef BAYESMIX_HIERARCHIES_UPDATERS_CONJUGATE_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_CONJUGATE_UPDATER_H_

#include "abstract_updater.h"
#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

template <class Likelihood, class PriorModel>
class ConjugateUpdater : public AbstractUpdater {
 public:
  ConjugateUpdater() = default;
  ~ConjugateUpdater() = default;

  bool is_conjugate() const override { return true; };
  void draw(AbstractLikelihood& like, AbstractPriorModel& prior,
            bool update_params) override;

 protected:
  Likelihood& downcast_likelihood(AbstractLikelihood& like_);
  PriorModel& downcast_prior(AbstractPriorModel& prior_);
};

// Methods' definitions
template <class Likelihood, class PriorModel>
Likelihood& ConjugateUpdater<Likelihood, PriorModel>::downcast_likelihood(
    AbstractLikelihood& like_) {
  return static_cast<Likelihood&>(like_);
}

template <class Likelihood, class PriorModel>
PriorModel& ConjugateUpdater<Likelihood, PriorModel>::downcast_prior(
    AbstractPriorModel& prior_) {
  return static_cast<PriorModel&>(prior_);
}

template <class Likelihood, class PriorModel>
void ConjugateUpdater<Likelihood, PriorModel>::draw(AbstractLikelihood& like,
                                                    AbstractPriorModel& prior,
                                                    bool update_params) {
  // Likelihood and PriorModel downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);

  // Sample from the full conditional of a conjugate hierarchy
  bool set_card = true, use_post_hypers=true;
  if (likecast.get_card() == 0) {
    likecast.set_state_from_proto(*priorcast.sample(!use_post_hypers), !set_card);
  } else {
    auto prev_hypers = priorcast.get_posterior_hypers();
    compute_posterior_hypers(likecast, priorcast);
    likecast.set_state_from_proto(*priorcast.sample(use_post_hypers), !set_card);
    if (!update_params)
      priorcast.set_posterior_hypers(prev_hypers);
  }
}

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_CONJUGATE_UPDATER_H_
