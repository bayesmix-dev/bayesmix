#ifndef BAYESMIX_SRC_HIERARCHIES_UPDATERS_TARGET_LPDF_UNCONSTRAINED_H_
#define BAYESMIX_SRC_HIERARCHIES_UPDATERS_TARGET_LPDF_UNCONSTRAINED_H_

#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

class target_lpdf_unconstrained {
 protected:
  AbstractLikelihood* like;
  AbstractPriorModel* prior;

 public:
  target_lpdf_unconstrained(AbstractLikelihood* like,
                            AbstractPriorModel* prior)
      : like(like), prior(prior) {}

  template <typename T>
  T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    return like->cluster_lpdf_from_unconstrained(x) +
           prior->lpdf_from_unconstrained(x);
  }
};

#endif
