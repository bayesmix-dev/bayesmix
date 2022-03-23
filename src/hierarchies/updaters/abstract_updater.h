#ifndef BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_

#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"
// #include "src/hierarchies/updaters/target_lpdf_unconstrained.h"

class AbstractUpdater {
 public:
  // Type alias
  using ProtoHypers = bayesmix::AlgorithmState::HierarchyHypers;

  //! Default destructor
  virtual ~AbstractUpdater() = default;

  //! Returns whether the current updater is for conjugate model or not
  virtual bool is_conjugate() const { return false; };

  //! Sampling from the full conditional, given the likelihood and the prior
  //! model that constitutes the hierarchy
  //! @param like The likelihood of the hierarchy
  //! @param prior The prior model of the hierarchy
  //! @param update_params Save posterior hyperparameters after draw?
  virtual void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
                    bool update_params) = 0;

  //! Computes the posterior hyperparameters required for the sampling in case
  //! of conjugate hierarchies
  virtual ProtoHypers compute_posterior_hypers(AbstractLikelihood &like,
                                               AbstractPriorModel &prior) {
    throw(std::runtime_error(
        "compute_posterior_hypers() not implemented for this updater"));
  }

  //! Stores the posterior hyperparameters in an appropriate container
  virtual void save_posterior_hypers(const ProtoHypers &post_hypers_) {
    throw(std::runtime_error(
        "save_posterior_hypers() not implemented for this updater"));
  }
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
