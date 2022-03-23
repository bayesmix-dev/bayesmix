#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/priors/nig_prior_model.h"

class NNIGUpdater
    : public SemiConjugateUpdater<UniNormLikelihood, NIGPriorModel> {
 public:
  NNIGUpdater() = default;
  ~NNIGUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypers compute_posterior_hypers(AbstractLikelihood &like,
                                       AbstractPriorModel &prior) override;

  // void compute_posterior_hypers(AbstractLikelihood& like,
  //                               AbstractPriorModel& prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNIG_UPDATER_H_
