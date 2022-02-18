#ifndef BAYESMIX_HIERARCHIES_UPDATERS_MALA_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_MALA_UPDATER_H_

#include <stan/math/rev.hpp>

#include "metropolis_updater.h"

class MalaUpdater : public MetropolisUpdater<MalaUpdater> {
 protected:
  double step_size;

 public:
  MalaUpdater() = default;
  ~MalaUpdater() = default;

  MalaUpdater(double step_size) : step_size(step_size) {}

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

  std::shared_ptr<MalaUpdater> clone() const {
    auto out =
        std::make_shared<MalaUpdater>(static_cast<MalaUpdater const &>(*this));
    return out;
  }
};

#endif
