#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNXIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNXIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/priors/nxig_prior_model.h"

//! Updater specific for the `UniNormLikelihood` used in combination
//! with `NxIGPriorModel`, that is the model
//! \f[
//!        y_i | \mu, \sigma^2 &\sim N(\mu, \sigma^2) \\
//!                     \mu &\sim N(\mu_0, \eta^2) \\
//!                  \sigma^2 & \sim IG(a,b)
//! \f]
//! It exploits the semi-conjugacy of the model to sample the full conditional
//! of (mu, sigsq) by calling `NxIGPriorModel::sample` with updated parameters

class NNxIGUpdater
    : public SemiConjugateUpdater<UniNormLikelihood, NxIGPriorModel> {
 public:
  NNxIGUpdater() = default;
  ~NNxIGUpdater() = default;

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNXIG_UPDATER_H_
