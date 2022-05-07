#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/multi_norm_likelihood.h"
#include "src/hierarchies/priors/nw_prior_model.h"

//! Updater specific for the `MultiNormLikelihood` used in combination
//! with `NWPriorModel`, that is the model
//!        y_i | mu, Sigma ~ Nd(mu, Sigma)
//!             mu | Sigma ~ N_d(mu0, sigsq / lambda)
//!             Sigma^{-1} ~ Wishart(nu, Psi)
//!
//! It exploits the conjugacy of the model to sample the full conditional of
//! (mu, sigsq) by calling `NWPriorModel::sample` with updated parameters

class NNWUpdater
    : public SemiConjugateUpdater<MultiNormLikelihood, NWPriorModel> {
 public:
  NNWUpdater() = default;
  ~NNWUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_
