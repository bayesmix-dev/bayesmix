#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_

#include "conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/priors/nig_prior_model.h"

class NNIGUpdater : public ConjugateUpdater<UniNormLikelihood, NIGPriorModel> {
 public:
  NNIGUpdater() = default;
  ~NNIGUpdater() = default;

  void initialize(AbstractLikelihood& like,
                  AbstractPriorModel& prior) override;
  void compute_posterior_hypers(AbstractLikelihood& like,
                                AbstractPriorModel& prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_
