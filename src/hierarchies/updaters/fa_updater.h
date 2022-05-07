#ifndef BAYESMIX_HIERARCHIES_UPDATERS_FA_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_FA_UPDATER_H_

#include "abstract_updater.h"
#include "src/hierarchies/likelihoods/fa_likelihood.h"
#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/fa_prior_model.h"
#include "src/hierarchies/priors/hyperparams.h"
#include "src/utils/proto_utils.h"

//! Updater specific for the `FAHierachy`.
//! See  Bhattacharya, Anirban, and David B. Dunson.
//! "Sparse Bayesian infinite factor models." Biometrika (2011): 291-306.
//! for further details

class FAUpdater : public AbstractUpdater {
 public:
  FAUpdater() = default;
  ~FAUpdater() = default;
  void draw(AbstractLikelihood& like, AbstractPriorModel& prior,
            bool update_params) override;

 protected:
  void sample_eta(State::FA& state, const Hyperparams::FA& hypers,
                  const FALikelihood& like);
  void sample_mu(State::FA& state, const Hyperparams::FA& hypers,
                 const FALikelihood& like);
  void sample_lambda(State::FA& state, const Hyperparams::FA& hypers,
                     const FALikelihood& like);
  void sample_psi(State::FA& state, const Hyperparams::FA& hypers,
                  const FALikelihood& like);
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_FA_UPDATER_H_
