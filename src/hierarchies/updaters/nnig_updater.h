#ifndef BAYESMIX_HIERARCHIES_NNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_NNIG_UPDATER_H_

#include "src/hierarchies/likelihoods/states.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/priors/hyperparams.h"
#include "src/hierarchies/priors/nig_prior_model.h"

class NNIGUpdater {
 public:
  NNIGUpdater() = default;
  ~NNIGUpdater() = default;

  std::shared_ptr<NNIGUpdater> clone() const;
  bool is_conjugate() const { return true; };
  void draw(UniNormLikelihood& like, NIGPriorModel& prior);
  void initialize(UniNormLikelihood& like, NIGPriorModel& prior);
  void compute_posterior_hypers(UniNormLikelihood& like, NIGPriorModel& prior);
};

#endif  // BAYESMIX_HIERARCHIES_NNIG_UPDATERS_H_
