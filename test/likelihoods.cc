#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "ls_state.pb.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/utils/rng.h"

TEST(uni_norm_likelihood, state_setget) {
  // Instance
  auto like = std::make_shared<UniNormLikelihood>();

  // Set state from proto
  bayesmix::UniLSState state_;
  bayesmix::AlgorithmState::ClusterState clust_state_;
  state_.set_mean(5.23);
  state_.set_var(1.02);
  clust_state_.mutable_uni_ls_state()->CopyFrom(state_);
  like->set_state_from_proto(clust_state_);

  // Get state proto
  auto out = like->get_state_proto();

  // Check if coincides
  ASSERT_EQ(out->DebugString(), clust_state_.DebugString());
}
