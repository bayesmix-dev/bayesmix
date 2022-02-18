#ifndef BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_

#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"
#include "src/hierarchies/updaters/target_lpdf_unconstrained.h"

class AbstractUpdater {
 public:
  virtual ~AbstractUpdater() = default;

  virtual bool is_conjugate() const { return false; };

  virtual void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
                    bool update_params) = 0;

  virtual void compute_posterior_hypers(AbstractLikelihood &like,
                                        AbstractPriorModel &prior) {
    throw std::runtime_error("compute_posterior_hypers not implemented");
  }
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
