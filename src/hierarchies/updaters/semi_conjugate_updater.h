#ifndef BAYESMIX_HIERARCHIES_UPDATERS_SEMI_CONJUGATE_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_SEMI_CONJUGATE_UPDATER_H_

#include <utility>

#include "abstract_updater.h"
#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

//! Updater for semi-conjugate hierarchies.
//!
//! We say that a hierarchy is semi-conjugate if the full conditionals
//! of each parameter is in the same parametric family of the prior
//! distribution of that parameter.
//!
//! As a consequence, sampling from the full conditional can be done
//! by calling the `sample` method from the `PriorModel` class, with
//! updater hyperparameters
//!
//! Classes inheriting from this one should only implement the
//! `compute_posterior_hypers(...)` member function
//!
//! This class is templated with respect to
//! @tparam Likelihood: the likelihood of the hierarchy, instance of
//! `AbstractLikelihood`
//! @tparam PriorModel: the prior of the hierarchy, instance of
//! `AbstractPriorModel`

template <class Likelihood, class PriorModel>
class SemiConjugateUpdater : public AbstractUpdater {
 public:
  SemiConjugateUpdater() = default;

  ~SemiConjugateUpdater() = default;

  void draw(AbstractLikelihood& like, AbstractPriorModel& prior,
            bool update_params) override;

  //! Used by algorithms such as Neal3 and SplitMerge
  //! It stores the hyperparameters computed by `compute_posterior_hypers`
 protected:
  Likelihood& downcast_likelihood(AbstractLikelihood& like_);
  PriorModel& downcast_prior(AbstractPriorModel& prior_);
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
  bool set_card = true;
  if (likecast.get_card() == 0) {
    likecast.set_state(priorcast.sample(), !set_card);
  } else {
    auto post_params =
        get_posterior_hypers(likecast, priorcast, update_params);
    likecast.set_state(priorcast.sample(post_params), !set_card);
  }
}

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_SEMI_CONJUGATE_UPDATER_H_
