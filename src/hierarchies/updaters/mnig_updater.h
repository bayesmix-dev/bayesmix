#ifndef BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_MNIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/uni_lin_reg_likelihood.h"
#include "src/hierarchies/priors/mnig_prior_model.h"

//! Updater specific for the `UniLinRegLikelihood` used in combination
//! with `MNIGPriorModel`, that is the model
//!        y_i | reg_coeffs, var ~ N(reg_coeffs^T x_i, var)
//!             reg_coeffs | var ~ N_p(mu0, sigsq * V^{-1})
//!                          var ~ InvGamma(a, b)
//!
//! It exploits the conjugacy of the model to sample the full conditional of
//! (reg_coeffs, var) by calling `MNIGPriorModel::sample` with updated
//! parameters
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
