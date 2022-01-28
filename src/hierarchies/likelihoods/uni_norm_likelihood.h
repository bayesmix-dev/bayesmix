#ifndef BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states.h"

class UniNormLikelihood
    : public BaseLikelihood<UniNormLikelihood, State::UniLS> {
 public:
  UniNormLikelihood() = default;
  ~UniNormLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return false; };
  void set_state_from_proto(const google::protobuf::Message &state_,
                            bool update_card = true) override;
  void clear_summary_statistics() override;
  double get_data_sum() const { return data_sum; };
  double get_data_sum_squares() const { return data_sum_squares; };

  template <typename T>
  T cluster_lpdf_from_unconstrained(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &unconstrained_params) const {
    assert(unconstrained_params.size() == 2);
    T mean = unconstrained_params(0);
    T var = stan::math::positive_constrain(unconstrained_params(1));
    T out = -(data_sum_squares - 2 * mean * data_sum + card * mean * mean) /
            (2 * var);
    out -= card * 0.5 * stan::math::log(stan::math::TWO_PI * var);
    return out;
  }

 protected:
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  double data_sum = 0;
  double data_sum_squares = 0;
};

#endif  // BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_
