#ifndef BAYESMIX_HIERARCHIES_UPDATERS_RANDOM_WALK_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_RANDOM_WALK_UPDATER_H_

#include "metropolis_updater.h"

/**
 * Metropolis-Hastings updater using an isotropic proposal function
 * centered in the current value of the parameters (unconstrained).
 * This class requires that the Hierarchy's state implements
 * the `get_unconstrained()`, `set_from_unconstrained()` and
 * `log_det_jac()` functions.
 *
 * Given the current value of the unconstrained parameters \f$ x \f$, a new
 * value is proposed from
 *
 * \f[
 *    x_{new} \sim N(x, step\_size \cdot I)
 * \f]
 *
 * and then either accepted (in which case the hierarchy's state is
 * set to \f$ x_{new} \f$) or rejected.
 */

class RandomWalkUpdater : public MetropolisUpdater<RandomWalkUpdater> {
 protected:
  double step_size;

 public:
  RandomWalkUpdater() = default;
  ~RandomWalkUpdater() = default;

  RandomWalkUpdater(double step_size) : step_size(step_size) {}

  //! Samples from the proposal distribution
  //! @param curr_state the current state (unconstrained parametrization)
  //! @param like instance of likelihood
  //! @param prior instance of prior
  //! @param target_lpdf either double or stan::math::var. Needed for
  //!         stan's automatic differentiation. It is not used here.
  template <typename F>
  Eigen::VectorXd sample_proposal(Eigen::VectorXd curr_state,
                                  AbstractLikelihood &like,
                                  AbstractPriorModel &prior, F &target_lpdf) {
    Eigen::VectorXd step(curr_state.size());
    auto &rng = bayesmix::Rng::Instance().get();
    for (int i = 0; i < curr_state.size(); i++) {
      step(i) = stan::math::normal_rng(0, step_size, rng);
    }
    return curr_state + step;
  }

  //! Evaluates the log probability density function of the proposal
  //! @param prop_state the proposed state (at which to evaluate the lpdf)
  //! @param curr_state the current state (unconstrained parametrization)
  //! @param like instance of likelihood
  //! @param prior instance of prior
  //! @param target_lpdf either double or stan::math::var. Needed for
  //!         stan's automatic differentiation. It is not used here.
  template <typename F>
  double proposal_lpdf(Eigen::VectorXd prop_state, Eigen::VectorXd curr_state,
                       AbstractLikelihood &like, AbstractPriorModel &prior,
                       F &target_lpdf) {
    double out;
    for (int i = 0; i < prop_state.size(); i++) {
      out += stan::math::normal_lpdf(prop_state(i), curr_state(i), step_size);
    }
    return out;
  }

  //! Returns a shared_ptr to a new instance of `this`
  std::shared_ptr<AbstractUpdater> clone() const override {
    auto out = std::make_shared<RandomWalkUpdater>(
        static_cast<RandomWalkUpdater const &>(*this));
    out->clear_hypers();
    return out;
  }
};

#endif
