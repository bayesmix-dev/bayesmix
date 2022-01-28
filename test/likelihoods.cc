#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "ls_state.pb.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
#include "src/hierarchies/likelihoods/multi_norm_likelihood.h"

#include "src/utils/proto_utils.h"
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

TEST(multi_norm_likelihood, set_get_state) {
  // Instance
  auto like = std::make_shared<MultiNormLikelihood>();

  // Prepare buffers
  bayesmix::MultiLSState state_;
  bayesmix::AlgorithmState::ClusterState set_state_;
  bayesmix::AlgorithmState::ClusterState got_state_;

  // Prepare state
  Eigen::Vector2d mu = {5.5, 5.5}; //mu << 5.5, 5.5;
  Eigen::Matrix2d prec = Eigen::Matrix2d::Identity();
  bayesmix::Vector mean_proto;
  bayesmix::Matrix prec_proto;
  bayesmix::to_proto(mu, &mean_proto);
  bayesmix::to_proto(prec, &prec_proto);
  set_state_.mutable_multi_ls_state()->mutable_mean()->CopyFrom(mean_proto);
  set_state_.mutable_multi_ls_state()->mutable_prec()->CopyFrom(prec_proto);
  set_state_.mutable_multi_ls_state()->mutable_prec_chol()->CopyFrom(prec_proto);

  // Set and get the state
  like->set_state_from_proto(set_state_);
  like->write_state_to_proto(&got_state_);

  // Check if they coincides
  ASSERT_EQ(got_state_.DebugString(), set_state_.DebugString());
}

TEST(multi_norm_likelihood, add_remove_data) {
  // Instance
  auto like = std::make_shared<MultiNormLikelihood>();

  // Add new datum to likelihood
  Eigen::RowVectorXd datum(2);
  datum << 5.5, 5.5;
  like->add_datum(0, datum);

  // Check if cardinality is augmented
  ASSERT_EQ(like->get_card(), 1);

  // Remove datum from likelihood
  like->remove_datum(0, datum);

  // Check if cardinality is reduced
  ASSERT_EQ(like->get_card(), 0);
}

TEST(multi_norm_likelihood, eval_lpdf) {
  // Instance
  auto like = std::make_shared<MultiNormLikelihood>();

  // Set state from proto
  bayesmix::AlgorithmState::ClusterState clust_state_;
  Eigen::Vector2d mu = {5.5, 5.5}; //mu << 5.5, 5.5;
  Eigen::Matrix2d prec = Eigen::Matrix2d::Identity();
  bayesmix::Vector mean_proto;
  bayesmix::Matrix prec_proto;
  bayesmix::to_proto(mu, &mean_proto);
  bayesmix::to_proto(prec, &prec_proto);
  clust_state_.mutable_multi_ls_state()->mutable_mean()->CopyFrom(mean_proto);
  clust_state_.mutable_multi_ls_state()->mutable_prec()->CopyFrom(prec_proto);
  clust_state_.mutable_multi_ls_state()->mutable_prec_chol()->CopyFrom(prec_proto);
  like->set_state_from_proto(clust_state_);

  // Data matrix on which evaluate the likelihood
  Eigen::MatrixXd data(3,2);
  data.row(0) << 4.5, 4.5;
  data.row(1) << 5.1, 5.1;
  data.row(2) << 2.5, 2.5;

  // Compute lpdf on this grid of points
  auto evals = like->lpdf_grid(data);
  auto like_copy = like->clone();
  auto evals_copy = like_copy->lpdf_grid(data);

  // Check if they coincides
  ASSERT_EQ(evals, evals_copy);
}