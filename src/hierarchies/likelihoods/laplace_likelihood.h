#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_LAPLACE_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_LAPLACE_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

class LaplaceLikelihood
    : public BaseLikelihood<LaplaceLikelihood, State::UniLS> {
 public:
  LaplaceLikelihood() = default;
  ~LaplaceLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return false; };
  void set_state_from_proto(const google::protobuf::Message &state_,
                            bool update_card = true) override;
  void clear_summary_statistics() override;

  template <typename T>
  T cluster_lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    assert(unconstrained_params.size() == 2);
    T mean = unconstrained_params(0);
    T var = stan::math::positive_constrain(unconstrained_params(1));
    T out = 0.;
    for (auto it = cluster_data_values.begin();
         it != cluster_data_values.end(); ++it) {
      out += stan::math::double_exponential_lpdf(*it, mean,
                                                 stan::math::sqrt(var / 2));
    }
    return out;
  }

  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  //! Set of values of data points belonging to this cluster
  std::list<Eigen::RowVectorXd> cluster_data_values;
  //! Sum of absolute differences for current params
  // double sum_abs_diff_curr = 0;
  //! Sum of absolute differences for proposal params
  // double sum_abs_diff_prop = 0;
};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_LAPLACE_LIKELIHOOD_H_
