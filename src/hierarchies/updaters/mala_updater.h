#ifndef BAYESMIX_HIERARCHIES_UPDATERS_MALA_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_MALA_UPDATER_H_

#include <stan/math/rev.hpp>

#include "metropolis_updater.h"

//! Metropolis Adjusted Langevin Algorithm.
//!
//! This class requires that the Hierarchy's state implements
//! the `get_unconstrained()`, `set_from_unconstrained()` and
//! `log_det_jac()` functions.
//!
//! Given the current value of the unconstrained parameters x, a new
//! value is proposed from
//!      x_new ~ N(x + step_size * grad(full_cond)(x), sqrt(2 step_size) * I)
//! and then either accepted (in which case the hierarchy's state is
//! set to x_new) or rejected.
class MalaUpdater : public MetropolisUpdater<MalaUpdater> {
 protected:
  double step_size;

 public:
  MalaUpdater() = default;
  ~MalaUpdater() = default;

  MalaUpdater(double step_size) : step_size(step_size) {}

  //! Samples from the proposal distribution
  //! @param curr_state the current state (unconstrained parametrization)
  //! @param like instance of likelihood
  //! @param prior instance of prior
  //! @param target_lpdf either double or stan::math::var. Needed for
  //!         stan's automatic differentiation. It will be
  //!         filled with the lpdf at the 'curr_state'
  Eigen::VectorXd sample_proposal(Eigen::VectorXd curr_state,
                                  AbstractLikelihood &like,
                                  AbstractPriorModel &prior,
                                  target_lpdf_unconstrained &target_lpdf) {
    Eigen::VectorXd noise(curr_state.size());
    auto &rng = bayesmix::Rng::Instance().get();
    double noise_scale = std::sqrt(2 * step_size);
    for (int i = 0; i < curr_state.size(); i++) {
      noise(i) = stan::math::normal_rng(0, noise_scale, rng);
    }
    Eigen::VectorXd grad;
    double tmp;
    stan::math::gradient(target_lpdf, curr_state, tmp, grad);
    return curr_state + step_size * grad + noise;
  }

  //! Evaluates the log probability density function of the proposal
  //! @param prop_state the proposed state (at which to evaluate the lpdf)
  //! @param curr_state the current state (unconstrained parametrization)
  //! @param like instance of likelihood
  //! @param prior instance of prior
  //! @param target_lpdf either double or stan::math::var. Needed for
  //!         stan's automatic differentiation. It will be
  //!         filled with the lpdf at 'curr_state'
  double proposal_lpdf(Eigen::VectorXd prop_state, Eigen::VectorXd curr_state,
                       AbstractLikelihood &like, AbstractPriorModel &prior,
                       target_lpdf_unconstrained &target_lpdf) {
    double out;
    Eigen::VectorXd grad;
    double tmp;
    stan::math::gradient(target_lpdf, curr_state, tmp, grad);
    Eigen::VectorXd mean = curr_state + step_size * grad;

    double noise_scale = std::sqrt(2 * step_size);

    for (int i = 0; i < prop_state.size(); i++) {
      out += stan::math::normal_lpdf(prop_state(i), mean(i), noise_scale);
    }
    return out;
  }

  //! Returns a shared_ptr to a new instance of `this`
  std::shared_ptr<MalaUpdater> clone() const {
    auto out =
        std::make_shared<MalaUpdater>(static_cast<MalaUpdater const &>(*this));
    return out;
  }
};

#endif
