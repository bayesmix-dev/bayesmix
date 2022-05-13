#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/priors/nig_prior_model.h"

//! Updater specific for the `UniNormLikelihood` used in combination
//! with `NIGPriorModel`, that is the model
//!        y_i | mu, sigsq ~ N(mu, sigsq)
//!             mu | sigsq ~ N(mu0, sigsq / lambda)
//!                  sigsq ~ InvGamma(a, b)
//!
//! It exploits the conjugacy of the model to sample the full conditional of
//! (mu, sigsq) by calling `NIGPriorModel::sample` with updated parameters

class NNIGUpdater
    : public SemiConjugateUpdater<UniNormLikelihood, NIGPriorModel> {
 public:
  NNIGUpdater() = default;
  ~NNIGUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_
