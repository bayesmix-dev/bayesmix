#ifndef BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_UNI_NORM_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states.h"

class UniNormLikelihood
    : public BaseLikelihood<UniNormLikelihood, State::UniLS> {
 private:
  double data_sum = 0;
  double data_sum_squares = 0;

 public:
  UniNormLikelihood() = default;

  ~UniNormLikelihood() = default;

  bool is_multivariate() const override { return false; };

  bool is_dependent() const override { return false; };

  void set_state_from_proto(const google::protobuf::Message &state_) override {
    auto &statecast = downcast_state(state_);
    state.mean = statecast.uni_ls_state().mean();
    state.var = statecast.uni_ls_state().var();
    set_card(statecast.cardinality());
  };

  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override {
    bayesmix::UniLSState state_;
    state_.set_mean(state.mean);
    state_.set_var(state.var);

    auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
    out->mutable_uni_ls_state()->CopyFrom(state_);
    return out;
  };

  void clear_summary_statistics() override {
    data_sum = 0;
    data_sum_squares = 0;
  }
};

#endif
