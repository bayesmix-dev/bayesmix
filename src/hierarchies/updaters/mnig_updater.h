#ifndef BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_lin_reg_likelihood.h"
#include "src/hierarchies/priors/mnig_prior_model.h"

/**
 * Updater specific for the `UniLinRegLikelihood` used in combination
 * with `MNIGPriorModel`, that is the model
 *
 * \f[
 *    y_i \mid \bm{\beta}, \sigma^2 &\stackrel{\small\mathrm{iid}}{\sim}
 * N(\bm{\beta}^T\bm{x}_i, \sigma^2) \\
 *  \bm{\beta} \mid \sigma^2 &\sim N_p(\mu_{0}, \sigma^2 \mathbf{V}^{-1}) \\
 *    \sigma^2 &\sim InvGamma(a, b)
 * \f]
 *
 * It exploits the conjugacy of the model to sample the full conditional of
 * \f$ (\bm{\beta}, \sigma^2) \f$ by calling `MNIGPriorModel::sample` with
 * updated parameters
 */

class MNIGUpdater
    : public SemiConjugateUpdater<UniLinRegLikelihood, MNIGPriorModel> {
 public:
  MNIGUpdater() = default;
  ~MNIGUpdater() = default;

  bool is_conjugate() const override { return true; };

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
