#ifndef BAYESMIX_HIERARCHIES_UPDATERS_METROPOLIS_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_METROPOLIS_UPDATER_H_

#include "abstract_updater.h"
#include "target_lpdf_unconstrained.h"

//! Base class for updaters using a Metropolis-Hastings algorithm
//!
//! This class serves as the base for a CRTP.
//! Children of this class should implement the methods
//!     template <typename F>
//!     Eigen::VectorXd sample_proposal(Eigen::VectorXd curr_state,
//!                                     AbstractLikelihood &like,
//!                                     AbstractPriorModel &prior, F
//!                                     &target_lpdf)
//! and
//!     template <typename F>
//!     double proposal_lpdf(Eigen::VectorXd prop_state, Eigen::VectorXd
//!     curr_state,
//!                          AbstractLikelihood &like, AbstractPriorModel
//!                          &prior,
//!                           F &target_lpdf)
//! where the template parameter is neeeded to allow the use of stan's
//! automatic differentiation if the gradient of the full conditional is
//! required.
template <class DerivedUpdater>
class MetropolisUpdater : public AbstractUpdater {
 public:
  //! Samples from the full conditional distribution using a
  //! Metropolis-Hasings step
  void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
            bool update_params) override {
    if (update_params) {
      throw std::runtime_error(
          "'update_params' can be True only when using instances of"
          " 'SemiConjugateUpdater'. This is likely caused by"
          " using a nonconjugate hierarchy (or a nonconjugate updater)"
          " in a marginal algorithm such as 'Neal3'.");
    }

    target_lpdf_unconstrained target_lpdf(&like, &prior);
    Eigen::VectorXd curr_state = like.get_unconstrained_state();
    Eigen::VectorXd prop_state =
        static_cast<DerivedUpdater *>(this)->sample_proposal(
            curr_state, like, prior, target_lpdf);

    double log_arate = like.cluster_lpdf_from_unconstrained(prop_state) -
                       like.cluster_lpdf_from_unconstrained(curr_state) +
                       static_cast<DerivedUpdater *>(this)->proposal_lpdf(
                           curr_state, prop_state, like, prior, target_lpdf) -
                       static_cast<DerivedUpdater *>(this)->proposal_lpdf(
                           prop_state, curr_state, like, prior, target_lpdf);

    auto &rng = bayesmix::Rng::Instance().get();
    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_arate) {
      like.set_state_from_unconstrained(prop_state);
    }
  }
};

#endif
