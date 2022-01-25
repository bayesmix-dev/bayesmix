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

TEST(uni_norm_likelihood, eval_lpdf_unconstrained) {
  // Instance
  auto like = std::make_shared<UniNormLikelihood>();

  // Set state from proto
  bayesmix::UniLSState state_;
  bayesmix::AlgorithmState::ClusterState clust_state_;
  double mean = 5;
  double var = 1;
  state_.set_mean(mean);
  state_.set_var(var);
  Eigen::VectorXd unconstrained_params(2);
  unconstrained_params << mean, std::log(var);
  clust_state_.mutable_uni_ls_state()->CopyFrom(state_);
  like->set_state_from_proto(clust_state_);

  // Add new datum to likelihood
  Eigen::VectorXd data(3);
  data << 4.5, 5.1, 2.5;
  double lpdf = 0.0;
  for (int i = 0; i < data.size(); ++i) {
    like->add_datum(i, data.row(i));
    lpdf += like->lpdf(data.row(i));
  }

  double clus_lpdf =
      like->cluster_lpdf_from_unconstrained(unconstrained_params);
  ASSERT_DOUBLE_EQ(lpdf, clus_lpdf);

  unconstrained_params(0) = 4.0;
  clus_lpdf = like->cluster_lpdf_from_unconstrained(unconstrained_params);
  ASSERT_TRUE(std::abs(clus_lpdf - lpdf) > 1e-5);
}
