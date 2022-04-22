#ifndef BAYESMIX_HIERARCHIES_GAMMA_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_GAMMA_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "src/hierarchies/likelihoods/base_likelihood.h"
#include "src/hierarchies/likelihoods/states/base_state.h"

namespace State {
class Gamma : public BaseState {
 public:
  double shape, rate;
  using ProtoState = bayesmix::AlgorithmState::ClusterState;

  ProtoState get_as_proto() const override {
    ProtoState out;
    out.mutable_general_state()->set_size(2);
    out.mutable_general_state()->mutable_data()->Add(shape);
    out.mutable_general_state()->mutable_data()->Add(rate);
    return out;
  }

  void set_from_proto(const ProtoState &state_, bool update_card) override {
    if (update_card) {
      card = state_.cardinality();
    }
    shape = state_.general_state().data()[0];
    rate = state_.general_state().data()[1];
  }
};
}  // namespace State

class GammaLikelihood : public BaseLikelihood<GammaLikelihood, State::Gamma> {
 public:
  GammaLikelihood() = default;
  ~GammaLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return false; };
  void clear_summary_statistics() override;

  // Getters and Setters
  int get_ndata() const { return ndata; };
  double get_shape() const { return state.shape; };
  double get_data_sum() const { return data_sum; };

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  //! Sum of data in the cluster
  double data_sum = 0;
  //! number of data in the cluster
  int ndata = 0;
};

/* DEFINITIONS */
void GammaLikelihood::clear_summary_statistics() {
  data_sum = 0;
  ndata = 0;
}

double GammaLikelihood::compute_lpdf(const Eigen::RowVectorXd &datum) const {
  return stan::math::gamma_lpdf(datum(0), state.shape, state.rate);
}

void GammaLikelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                       bool add) {
  if (add) {
    data_sum += datum(0);
    ndata += 1;
  } else {
    data_sum -= datum(0);
    ndata -= 1;
  }
}

#endif  // BAYESMIX_HIERARCHIES_GAMMA_LIKELIHOOD_H_
