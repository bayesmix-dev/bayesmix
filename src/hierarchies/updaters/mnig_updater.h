#ifndef BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_

#include "conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_lin_reg_likelihood.h"
#include "src/hierarchies/priors/mnig_prior_model.h"

class MNIGUpdater
    : public ConjugateUpdater<UniLinRegLikelihood, MNIGPriorModel> {
 public:
  MNIGUpdater() = default;
  ~MNIGUpdater() = default;

  void compute_posterior_hypers(AbstractLikelihood& like,
                                AbstractPriorModel& prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
