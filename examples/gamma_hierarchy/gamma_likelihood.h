#ifndef BAYESMIX_HIERARCHIES_GAMMA_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_GAMMA_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "src/hierarchies/likelihoods/base_likelihood.h"

namespace State {
class Gamma {
 public:
  double shape, rate;
};
}  // namespace State

class GammaLikelihood : public BaseLikelihood<GammaLikelihood, State::Gamma> {
 public:
  GammaLikelihood() = default;
  ~GammaLikelihood() = default;
  bool is_multivariate() const override { return false; };
  bool is_dependent() const override { return false; };
  void set_state_from_proto(const google::protobuf::Message &state_,
                            bool update_card = true) override;
  void clear_summary_statistics() override;

  // Getters and Setters
  int get_ndata() const { return ndata; };
  double get_shape() const { return state.shape; };
  double get_data_sum() const { return data_sum; };
  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override;

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  //! Sum of data in the cluster
  double data_sum = 0;
  //! number of data in the cluster
  int ndata = 0;
};

/* DEFINITIONS */
void GammaLikelihood::set_state_from_proto(
    const google::protobuf::Message &state_, bool update_card) {
  auto &statecast = downcast_state(state_);
  state.rate = statecast.general_state().data()[0];
  if (update_card) set_card(statecast.cardinality());
}

void GammaLikelihood::clear_summary_statistics() {
  data_sum = 0;
  ndata = 0;
}

std::shared_ptr<bayesmix::AlgorithmState::ClusterState>
GammaLikelihood::get_state_proto() const {
  bayesmix::Vector state_;
  state_.mutable_data()->Add(state.shape);
  state_.mutable_data()->Add(state.rate);

  auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
  out->mutable_general_state()->CopyFrom(state_);
  return out;
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
