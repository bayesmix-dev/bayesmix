#ifndef BAYESMIX_SRC_HIERARCHIES_UPDATERS_TARGET_LPDF_UNCONSTRAINED_H_
#define BAYESMIX_SRC_HIERARCHIES_UPDATERS_TARGET_LPDF_UNCONSTRAINED_H_

#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

//! Functor that computes the log-full conditional distribution
//! of a specific hierarchy.
//! Used by metropolis-like updaters especially when the gradient
//! of the target_lpdf if required
class target_lpdf_unconstrained {
 protected:
  AbstractLikelihood* like;
  AbstractPriorModel* prior;

 public:
  target_lpdf_unconstrained(AbstractLikelihood* like,
                            AbstractPriorModel* prior)
      : like(like), prior(prior) {}

  //! Computes the log-full conditional that is simply the
  //! sum of `cluster_lpdf_from_unconstrained` in `AbstractLikelihood`
  //! and `lpdf_from_unconstrained` in `AbstractPriorModel`
  template <typename T>
  T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    return like->cluster_lpdf_from_unconstrained(x) +
           prior->lpdf_from_unconstrained(x);
  }
};

#endif
