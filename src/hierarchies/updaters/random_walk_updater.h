#ifndef BAYESMIX_HIERARCHIES_UPDATERS_RANDOM_WALK_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_RANDOM_WALK_UPDATER_H_

#include "metropolis_updater.h"

class RandomWalkUpdater : public MetropolisUpdater {
 protected:
  double step_size;

 public:
  RandomWalkUpdater() = default;
  ~RandomWalkUpdater() = default;

  RandomWalkUpdater(double step_size) : step_size(step_size) {}

  Eigen::VectorXd sample_proposal(Eigen::VectorXd curr_state,
                                  AbstractLikelihood &like,
                                  AbstractPriorModel &prior) override {
    Eigen::VectorXd step(curr_state.size());
    auto &rng = bayesmix::Rng::Instance().get();
    for (int i = 0; i < curr_state.size(); i++) {
      step(i) = stan::math::normal_rng(0, step_size, rng);
    }
    return curr_state + step;
  }

  double proposal_lpdf(Eigen::VectorXd prop_state, Eigen::VectorXd curr_state,
                       AbstractLikelihood &like,
                       AbstractPriorModel &prior) override {
    double out;
    for (int i = 0; i < prop_state.size(); i++) {
      out += stan::math::normal_lpdf(prop_state(i), curr_state(i), step_size);
    }
    return out;
  }

  std::shared_ptr<RandomWalkUpdater> clone() const {
    auto out = std::make_shared<RandomWalkUpdater>(
        static_cast<RandomWalkUpdater const &>(*this));
    return out;
  }
};

#endif
