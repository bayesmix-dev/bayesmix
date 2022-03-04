#include <gtest/gtest.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "ls_state.pb.h"
#include "src/hierarchies/likelihoods/laplace_likelihood.h"
#include "src/hierarchies/likelihoods/multi_norm_likelihood.h"
#include "src/hierarchies/likelihoods/uni_lin_reg_likelihood.h"
#include "src/hierarchies/likelihoods/uni_norm_likelihood.h"
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

TEST(multi_ls_state, set_unconstrained) {
  auto& rng = bayesmix::Rng::Instance().get();

  State::MultiLS state;
  auto mean = Eigen::VectorXd::Zero(5);
  auto prec =
      stan::math::wishart_rng(10, Eigen::MatrixXd::Identity(5, 5), rng);
  state.mean = mean;
  state.prec = prec;
  Eigen::VectorXd unconstrained_state = state.get_unconstrained();

  State::MultiLS state2;
  state2.set_from_unconstrained(unconstrained_state);
  ASSERT_TRUE((state.mean - state2.mean).squaredNorm() < 1e-5);
  ASSERT_TRUE((state.prec - state2.prec).squaredNorm() < 1e-5);
}

TEST(multi_norm_likelihood, set_get_state) {
  // Instance
  auto like = std::make_shared<MultiNormLikelihood>();

  // Prepare buffers
  bayesmix::MultiLSState state_;
  bayesmix::AlgorithmState::ClusterState set_state_;
  bayesmix::AlgorithmState::ClusterState got_state_;

  // Prepare state
  Eigen::Vector2d mu = {5.5, 5.5};  // mu << 5.5, 5.5;
  Eigen::Matrix2d prec = Eigen::Matrix2d::Identity();
  bayesmix::Vector mean_proto;
  bayesmix::Matrix prec_proto;
  bayesmix::to_proto(mu, &mean_proto);
  bayesmix::to_proto(prec, &prec_proto);
  set_state_.mutable_multi_ls_state()->mutable_mean()->CopyFrom(mean_proto);
  set_state_.mutable_multi_ls_state()->mutable_prec()->CopyFrom(prec_proto);
  set_state_.mutable_multi_ls_state()->mutable_prec_chol()->CopyFrom(
      prec_proto);

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
  Eigen::Vector2d mu = {5.5, 5.5};  // mu << 5.5, 5.5;
  Eigen::Matrix2d prec = Eigen::Matrix2d::Identity();
  bayesmix::Vector mean_proto;
  bayesmix::Matrix prec_proto;
  bayesmix::to_proto(mu, &mean_proto);
  bayesmix::to_proto(prec, &prec_proto);
  clust_state_.mutable_multi_ls_state()->mutable_mean()->CopyFrom(mean_proto);
  clust_state_.mutable_multi_ls_state()->mutable_prec()->CopyFrom(prec_proto);
  clust_state_.mutable_multi_ls_state()->mutable_prec_chol()->CopyFrom(
      prec_proto);
  like->set_state_from_proto(clust_state_);

  // Data matrix on which evaluate the likelihood
  Eigen::MatrixXd data(3, 2);
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

TEST(uni_lin_reg_likelihood, set_get_state) {
  // Instance
  auto like = std::make_shared<UniLinRegLikelihood>();

  // Prepare buffers
  bayesmix::LinRegUniLSState state_;
  bayesmix::AlgorithmState::ClusterState set_state_;
  bayesmix::AlgorithmState::ClusterState got_state_;

  // Prepare state
  Eigen::Vector3d reg_coeffs;
  reg_coeffs << 2.25, 0.22, -7.1;
  bayesmix::Vector reg_coeffs_proto;
  bayesmix::to_proto(reg_coeffs, &reg_coeffs_proto);
  state_.mutable_regression_coeffs()->CopyFrom(reg_coeffs_proto);
  state_.set_var(1.02);
  set_state_.mutable_lin_reg_uni_ls_state()->CopyFrom(state_);

  // Set and get the state
  like->set_state_from_proto(set_state_);
  like->write_state_to_proto(&got_state_);

  // Check if they coincides
  ASSERT_EQ(got_state_.DebugString(), set_state_.DebugString());
}

TEST(uni_lin_reg_likelihood, add_remove_data) {
  // Instance
  auto like = std::make_shared<UniLinRegLikelihood>();

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

TEST(uni_lin_reg_likelihood, eval_lpdf) {
  // Instance
  auto like = std::make_shared<UniLinRegLikelihood>();

  // Set state from proto
  bayesmix::LinRegUniLSState state_;
  bayesmix::AlgorithmState::ClusterState clust_state_;
  Eigen::Vector3d reg_coeffs;
  reg_coeffs << 2.25, 0.22, -7.1;
  bayesmix::Vector reg_coeffs_proto;
  bayesmix::to_proto(reg_coeffs, &reg_coeffs_proto);
  state_.mutable_regression_coeffs()->CopyFrom(reg_coeffs_proto);
  state_.set_var(1.02);
  clust_state_.mutable_lin_reg_uni_ls_state()->CopyFrom(state_);
  like->set_state_from_proto(clust_state_);

  // Generate data
  Eigen::Vector3d data;
  data << 4.5, 5.1, 2.5;

  // Generate random covariate matrix
  Eigen::MatrixXd cov =
      Eigen::MatrixXd::Random(data.size(), reg_coeffs.size());

  // Compute lpdf on this grid of points
  auto evals = like->lpdf_grid(data, cov);
  auto like_copy = like->clone();
  auto evals_copy = like_copy->lpdf_grid(data, cov);

  // Check if they coincides
  ASSERT_EQ(evals, evals_copy);
}

TEST(laplace_likelihood, set_get_state) {
  // Instance
  auto like = std::make_shared<LaplaceLikelihood>();

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

TEST(laplace_likelihood, add_remove_data) {
  // Instance
  auto like = std::make_shared<LaplaceLikelihood>();

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

TEST(laplace_likelihood, eval_lpdf) {
  // Instance
  auto like = std::make_shared<LaplaceLikelihood>();

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

TEST(laplace_likelihood, eval_lpdf_unconstrained) {
  // Instance
  auto like = std::make_shared<LaplaceLikelihood>();

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

  unconstrained_params(0) = 3.0;
  clus_lpdf = like->cluster_lpdf_from_unconstrained(unconstrained_params);
  ASSERT_TRUE(std::abs(clus_lpdf - lpdf) > 1e-5);
}
