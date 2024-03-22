#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_BETA_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_BETA_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

/**
 * A univariate Beta likelihood, using the `State::ShapeRate` state. Represents
 * the model:
 *
 * \f[
 *    y_1,\dots,y_k \mid \mu, \sigma^2 \stackrel{\small\mathrm{iid}}{\sim}
 * Beta(a, b),
 * \f]
 */

class BetaLikelihood
    : public BaseLikelihood<BetaLikelihood, State::ShapeRate> {
 public:
  BetaLikelihood() = default;
  ~BetaLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return false; };
  void clear_summary_statistics() override;

  // Eigen::VectorXd sample() const override;

  template <typename T>
  T cluster_lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    assert(unconstrained_params.size() == 2);

    T a = stan::math::positive_constrain(unconstrained_params(0));
    T b = stan::math::positive_constrain(unconstrained_params(1));

    T out = 0.;

    return (a - 1.) * sum_logs + (b - 1.) * sum_logs1m -
           card * (stan::math::lgamma(a) + stan::math::lgamma(b) -
                   stan::math::lgamma(a + b));
  }

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  double sum_logs = 0;
  double sum_logs1m = 0;
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_BETA_LIKELIHOOD_H_
