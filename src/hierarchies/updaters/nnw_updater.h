#ifndef BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/multi_norm_likelihood.h"
#include "src/hierarchies/priors/nw_prior_model.h"

/**
 * Updater specific for the `MultiNormLikelihood` used in combination
 * with `NWPriorModel`, that is the model
 *
 * \f[
 *      y_i \mid \bm{\mu}, \Sigma &\stackrel{\small\mathrm{iid}}{\sim}
 * N_d(\bm{mu}, \Sigma) \\
 *      \bm{\mu} \mid \Sigma &\sim N_d(\bm{\mu}_0, \Sigma / \lambda) \\
 *      \Sigma^{-1} &\sim Wishart(\nu, \Psi)
 * \f]
 *
 * It exploits the conjugacy of the model to sample the full conditional of
 * \f$ (\bm{\mu}, \Sigma) \f$ by calling `NWPriorModel::sample` with updated
 * parameters.
 */

class NNWUpdater
    : public SemiConjugateUpdater<MultiNormLikelihood, NWPriorModel> {
 public:
  NNWUpdater() = default;
  ~NNWUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;

  std::shared_ptr<AbstractUpdater> clone() const override {
    auto out =
        std::make_shared<NNWUpdater>(static_cast<NNWUpdater const &>(*this));
    out->clear_hypers();
    return out;
  }
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_NNW_UPDATER_H_
