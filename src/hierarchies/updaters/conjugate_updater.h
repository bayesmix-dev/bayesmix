#ifndef BAYESMIX_HIERARCHIES_UPDATERS_CONJUGATE_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_CONJUGATE_UPDATER_H_

// NOT WORKING AT THE MOMENT

#include "abstract_updater.h"

class ConjugateUpdater : public AbstractUpdater {
 public:
  ConjugateUpdater() = default;
  ~ConjugateUpdater() = default;
  bool is_conjugate() const override { return true; };
  void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
            bool update_params) override;
  virtual void compute_posterior_hypers(UniNormLikelihood &like,
                                        NIGPriorModel &prior) = 0;
};

void ConjugateUpdater::draw(AbstractLikelihood &like,
                            AbstractPriorModel &prior, bool update_params) {
  bool set_card = true;
  if (like.get_card() == 0) {
    like.set_state_from_proto(*prior.sample(false), !set_card);
  } else {
    if (update_params) {
      compute_posterior_hypers(like, prior);
    }
    like.set_state_from_proto(*prior.sample(true), !set_card);
  }
}

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_CONJUGATE_UPDATER_H_
