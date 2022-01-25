#ifndef BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>
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

  // The unconstrained parameters are mean and log(var)
  double cluster_lpdf_from_unconstrained(
      Eigen::VectorXd unconstrained_params) override;

 protected:
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  double data_sum = 0;
  double data_sum_squares = 0;
};

#endif  // BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_
