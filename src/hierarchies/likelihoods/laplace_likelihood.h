#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_LAPLACE_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_LAPLACE_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

/**
 * A univariate Laplace likelihood, using the `State::UniLS` state. Represents
 * the model:
 *
 * \f[
 *    y_1,\dots,y_k \mid \mu, \sigma^2 \stackrel{\small\mathrm{iid}}{\sim}
 * Laplace(\mu,\sigma^2), \f]
 *
 * where \f$ \mu \f$ is the mean and center of the distribution
 * and \f$ \sigma^2 \f$ is the variance. The scale parameter \f$ \lambda \f$ is
 * then \f$ \sqrt{\sigma^2/2} \f$. These parameters are stored in a
 * `State::UniLS` state. Since the Laplace likelihood does not have sufficient
 * statistics other than the whole sample, the `update_sum_stats()` method does
 * nothing.
 */

class LaplaceLikelihood
    : public BaseLikelihood<LaplaceLikelihood, State::UniLS> {
 public:
  LaplaceLikelihood() = default;
  ~LaplaceLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return false; };
  void clear_summary_statistics() override { return; };

  template <typename T>
  T cluster_lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    assert(unconstrained_params.size() == 2);

    T mean = unconstrained_params(0);
    T var = stan::math::positive_constrain(unconstrained_params(1));

    T out = 0.;
    for (auto it = cluster_data_idx.begin(); it != cluster_data_idx.end();
         ++it) {
      out += stan::math::double_exponential_lpdf(dataset_ptr->row(*it), mean,
                                                 stan::math::sqrt(var / 2.0));
    }
    return out;
  }

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override {
    return;
  };
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_LAPLACE_LIKELIHOOD_H_
