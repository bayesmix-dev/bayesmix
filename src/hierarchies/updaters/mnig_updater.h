#ifndef BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_lin_reg_likelihood.h"
#include "src/hierarchies/priors/mnig_prior_model.h"

class MNIGUpdater
    : public SemiConjugateUpdater<UniLinRegLikelihood, MNIGPriorModel> {
 public:
  MNIGUpdater() = default;
  ~MNIGUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;

  // void compute_posterior_hypers(AbstractLikelihood& like,
  //                               AbstractPriorModel& prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
