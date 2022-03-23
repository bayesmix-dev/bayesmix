#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/multi_norm_likelihood.h"
#include "src/hierarchies/priors/nw_prior_model.h"

class NNWUpdater
    : public SemiConjugateUpdater<MultiNormLikelihood, NWPriorModel> {
 public:
  NNWUpdater() = default;
  ~NNWUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypers compute_posterior_hypers(AbstractLikelihood &like,
                                       AbstractPriorModel &prior) override;

  // void compute_posterior_hypers(AbstractLikelihood& like,
  //                               AbstractPriorModel& prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_
