#ifndef BAYESMIX_HIERARCHIES_UPDATERS_METROPOLIS_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_METROPOLIS_UPDATER_H_

#include "abstract_updater.h"

class MetropolisUpdater : public AbstractUpdater {
 public:
  void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
            bool update_params) override {
    Eigen::VectorXd curr_state = like.get_unconstrained_state();
    Eigen::VectorXd prop_state = sample_proposal(curr_state, like, prior);

    double log_arate = like.cluster_lpdf_from_unconstrained(prop_state) -
                       like.cluster_lpdf_from_unconstrained(curr_state) +
                       proposal_lpdf(curr_state, prop_state, like, prior) -
                       proposal_lpdf(prop_state, curr_state, like, prior);

    auto &rng = bayesmix::Rng::Instance().get();
    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_arate) {
      like.set_state_from_unconstrained(prop_state);
    }
  }

  virtual Eigen::VectorXd sample_proposal(Eigen::VectorXd curr_state,
                                          AbstractLikelihood &like,
                                          AbstractPriorModel &prior) = 0;

  virtual double proposal_lpdf(Eigen::VectorXd prop_state,
                               Eigen::VectorXd curr_state,
                               AbstractLikelihood &like,
                               AbstractPriorModel &prior) = 0;
};

#endif
