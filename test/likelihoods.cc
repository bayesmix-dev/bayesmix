#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "ls_state.pb.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/utils/rng.h"

TEST(uni_norm_likelihood, set_get_state) {
  // Instance
  auto like = std::make_shared<UniNormLikelihood>();

  // Prepare buffers
  bayesmix::UniLSState state_;
  bayesmix::AlgorithmState::ClusterState set_state_;
  bayesmix::AlgorithmState::ClusterState got_state_;

  // Prepare state
  state_.set_mean(5.23);
  state_.set_var(1.02);
  set_state_.mutable_uni_ls_state()->CopyFrom(state_);

  // Set and get the state
  like->set_state_from_proto(set_state_);
  like->write_state_to_proto(&got_state_);

  // Check if they coincides
  ASSERT_EQ(got_state_.DebugString(), set_state_.DebugString());
}

TEST(uni_norm_likelihood, add_remove_data) {
  // Instance
  auto like = std::make_shared<UniNormLikelihood>();

  // Add new datum to likelihood
  Eigen::VectorXd datum(1);
  datum << 5.0;
  like->add_datum(0, datum);

  // Check if cardinality is augmented
  ASSERT_EQ(like->get_card(), 1);

  // Remove datum from likelihood
  like->remove_datum(0, datum);

  // Check if cardinality is reduced
  ASSERT_EQ(like->get_card(), 0);
}

TEST(uni_norm_likelihood, eval_lpdf) {
  // Instance
  auto like = std::make_shared<UniNormLikelihood>();

  // Set state from proto
  bayesmix::UniLSState state_;
  bayesmix::AlgorithmState::ClusterState clust_state_;
  state_.set_mean(5);
  state_.set_var(1);
  clust_state_.mutable_uni_ls_state()->CopyFrom(state_);
  like->set_state_from_proto(clust_state_);

  // Add new datum to likelihood
  Eigen::VectorXd data(3);
  data << 4.5, 5.1, 2.5;

  // Compute lpdf on this grid of points
  auto evals = like->lpdf_grid(data);
  auto like_copy = like->clone();
  auto evals_copy = like_copy->lpdf_grid(data);

  // Check if they coincides
  ASSERT_EQ(evals, evals_copy);
}
