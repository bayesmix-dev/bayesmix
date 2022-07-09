#ifndef BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_

#include "src/hierarchies/likelihoods/abstract_likelihood.h"
#include "src/hierarchies/priors/abstract_prior_model.h"

//! Base class for an Updater

//! An updater is a class able to sample from the full conditional
//! distribution of an `Hierarchy`, coming from the product of a
//! `Likelihood` and a `Prior`, possibly using a Metropolis-Hastings
//! algorithm.
class AbstractUpdater {
 public:
  // Type aliases
  using ProtoHypersPtr =
      std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers>;
  using ProtoHypers = ProtoHypersPtr::element_type;

  //! Default destructor
  virtual ~AbstractUpdater() = default;

  //! Returns whether the current updater is for a (semi)conjugate model or not
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
  virtual ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                                  AbstractPriorModel &prior) {
    if (!is_conjugate()) {
      throw(
          std::runtime_error("Cannot call compute_posterior_hypers() from a "
                             "non-(semi)conjugate updater"));
    } else {
      throw(std::runtime_error(
          "compute_posterior_hypers() not implemented for this updater"));
    }
  }

  //! Stores the posterior hyperparameters in an appropriate container
  void save_posterior_hypers(ProtoHypersPtr post_hypers_) {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Cannot call save_posterior_hypers() from a "
          "non-(semi)conjugate updater");
    } else {
      saved_posterior_hypers = true;
      posterior_hypers = post_hypers_;
    }
  }

  virtual ProtoHypersPtr get_posterior_hypers(AbstractLikelihood &like,
                                              AbstractPriorModel &prior) {
    if (!is_conjugate()) {
      throw std::runtime_error(
          "Cannot call get_posterior_hypers() from a "
          "non-(semi)conjugate updater");
    } else {
      if (!saved_posterior_hypers) {
        std::cout << "computing posterior hypers" << std::endl;
        posterior_hypers = compute_posterior_hypers(like, prior);
      }
      // std::cout << "saved posterior hypers " << std::endl;
      // posterior_hypers->PrintDebugString();

      // std::cout << "computed posterior hypers " << std::endl;
      // posterior_hypers = compute_posterior_hypers(like, prior);
      // posterior_hypers->PrintDebugString();

      // std::cout << std::endl << std::endl;

      return posterior_hypers;
    }
  }

  void clear_hypers() {
    saved_posterior_hypers = false;
    posterior_hypers = nullptr;
  }

  virtual std::shared_ptr<AbstractUpdater> clone() const = 0;

 protected:
  bool saved_posterior_hypers = false;
  ProtoHypersPtr posterior_hypers = nullptr;
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_ABSTRACT_UPDATER_H_
