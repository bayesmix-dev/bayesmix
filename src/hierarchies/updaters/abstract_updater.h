#ifndef BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_

// NOT WORKING AT THE MOMENT

#include <memory>

#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

class AbstractUpdater {
 public:
  virtual ~AbstractUpdater() = default;
  // virtual std::shared_ptr<AbstractUpdater> clone() const = 0; NON CREDO CI
  // SERVA
  bool is_conjugate() const { return false; };
  virtual void initialize(AbstractLikelihood &like,
                          AbstractPriorModel &prior) = 0;
  virtual void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
                    bool update_params) = 0;
  virtual void compute_posterior_hypers(UniNormLikelihood &like,
                                NIGPriorModel &prior) {
    throw std::runtime_error("compute_posterior_hypers not implemented");
  }
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
