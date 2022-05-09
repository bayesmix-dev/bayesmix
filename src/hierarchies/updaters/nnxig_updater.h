#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNXIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNXIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/priors/nxig_prior_model.h"

class NNxIGUpdater
    : public SemiConjugateUpdater<UniNormLikelihood, NxIGPriorModel> {
 public:
  NNxIGUpdater() = default;
  ~NNxIGUpdater() = default;

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNXIG_UPDATER_H_
