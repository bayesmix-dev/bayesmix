#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "src/hierarchies/priors/nig_prior_model.h"
#include "src/hierarchies/priors/nxig_prior_model.h"

TEST(nig_prior_model, set_get_hypers) {
  // Instance
  auto prior = std::make_shared<NIGPriorModel>();

  // Prepare buffers
  bayesmix::NIGDistribution hypers_;
  bayesmix::AlgorithmState::HierarchyHypers set_state_;
  bayesmix::AlgorithmState::HierarchyHypers got_state_;

  // Prepare hypers
  hypers_.set_mean(5.0);
  hypers_.set_var_scaling(0.1);
  hypers_.set_shape(4.0);
  hypers_.set_scale(3.0);
  set_state_.mutable_nnig_state()->CopyFrom(hypers_);

  // Set and get hypers
  prior->set_hypers_from_proto(set_state_);
  prior->write_hypers_to_proto(&got_state_);

  // Check if they coincides
  ASSERT_EQ(got_state_.DebugString(), set_state_.DebugString());
}

TEST(nig_prior_model, fixed_values_prior) {
  // Prepare buffers
  bayesmix::NNIGPrior prior;
  bayesmix::AlgorithmState::HierarchyHypers prior_out;
  std::vector<std::shared_ptr<AbstractPriorModel>> prior_models;
  std::vector<bayesmix::AlgorithmState::ClusterState> states;

  // Set fixed value prior
  prior.mutable_fixed_values()->set_mean(5.0);
  prior.mutable_fixed_values()->set_var_scaling(0.1);
  prior.mutable_fixed_values()->set_shape(2.0);
  prior.mutable_fixed_values()->set_scale(2.0);

  // Initialize prior model
  auto prior_model = std::make_shared<NIGPriorModel>();
  prior_model->get_mutable_prior()->CopyFrom(prior);
  prior_model->initialize();

  // Check equality before update
  prior_models.push_back(prior_model);
  for (size_t i = 1; i < 4; i++) {
    prior_models.push_back(prior_model->clone());
    prior_models[i]->write_hypers_to_proto(&prior_out);
    ASSERT_EQ(prior.fixed_values().DebugString(),
              prior_out.nnig_state().DebugString());
  }

  // Check equality after update
  prior_models[0]->update_hypers(states);
  prior_models[0]->write_hypers_to_proto(&prior_out);
  for (size_t i = 1; i < 4; i++) {
    prior_models[i]->write_hypers_to_proto(&prior_out);
    ASSERT_EQ(prior.fixed_values().DebugString(),
              prior_out.nnig_state().DebugString());
  }
}

TEST(nig_prior_model, normal_mean_prior) {
  // Prepare buffers
  bayesmix::NNIGPrior prior;
  bayesmix::AlgorithmState::HierarchyHypers prior_out;

  // Set Normal prior on the mean
  double mu00 = 0.5;
  prior.mutable_normal_mean_prior()->mutable_mean_prior()->set_mean(mu00);
  prior.mutable_normal_mean_prior()->mutable_mean_prior()->set_var(1.02);
  prior.mutable_normal_mean_prior()->set_var_scaling(0.1);
  prior.mutable_normal_mean_prior()->set_shape(2.0);
  prior.mutable_normal_mean_prior()->set_scale(2.0);

  // Prepare some fictional states
  std::vector<bayesmix::AlgorithmState::ClusterState> states(4);
  for (int i = 0; i < states.size(); i++) {
    double mean = 9.0 + i;
    states[i].mutable_uni_ls_state()->set_mean(mean);
    states[i].mutable_uni_ls_state()->set_var(1.0);
  }

  // Initialize prior model
  auto prior_model = std::make_shared<NIGPriorModel>();
  prior_model->get_mutable_prior()->CopyFrom(prior);
  prior_model->initialize();

  // Update hypers in light of current states
  prior_model->update_hypers(states);
  prior_model->write_hypers_to_proto(&prior_out);
  double mean_out = prior_out.nnig_state().mean();

  // Check
  ASSERT_GT(mean_out, mu00);
}

TEST(nig_prior_model, sample) {
  // Instance
  auto prior = std::make_shared<NIGPriorModel>();
  bool use_post_hypers = true;

  // Define prior hypers
  bayesmix::AlgorithmState::HierarchyHypers hypers_proto;
  hypers_proto.mutable_nnig_state()->set_mean(5.0);
  hypers_proto.mutable_nnig_state()->set_var_scaling(0.1);
  hypers_proto.mutable_nnig_state()->set_shape(4.0);
  hypers_proto.mutable_nnig_state()->set_scale(3.0);

  // Set hypers and get sampled state as proto
  prior->set_hypers_from_proto(hypers_proto);
  auto state1 = prior->sample(!use_post_hypers);
  auto state2 = prior->sample(!use_post_hypers);

  // Check if they coincides
  ASSERT_TRUE(state1->DebugString() != state2->DebugString());
}

TEST(nxig_prior_model, set_get_hypers) {
  // Instance
  auto prior = std::make_shared<NxIGPriorModel>();

  // Prepare buffers
  bayesmix::NxIGDistribution hypers_;
  bayesmix::AlgorithmState::HierarchyHypers set_state_;
  bayesmix::AlgorithmState::HierarchyHypers got_state_;

  // Prepare hypers
  hypers_.set_mean(5.0);
  hypers_.set_var(1.2);
  hypers_.set_shape(4.0);
  hypers_.set_scale(3.0);
  set_state_.mutable_nnxig_state()->CopyFrom(hypers_);

  // Set and get hypers
  prior->set_hypers_from_proto(set_state_);
  prior->write_hypers_to_proto(&got_state_);

  // Check if they coincides
  ASSERT_EQ(got_state_.DebugString(), set_state_.DebugString());
}

TEST(nxig_prior_model, fixed_values_prior) {
  // Prepare buffers
  bayesmix::NNxIGPrior prior;
  bayesmix::AlgorithmState::HierarchyHypers prior_out;
  std::vector<std::shared_ptr<AbstractPriorModel>> prior_models;
  std::vector<bayesmix::AlgorithmState::ClusterState> states;

  // Set fixed value prior
  prior.mutable_fixed_values()->set_mean(5.0);
  prior.mutable_fixed_values()->set_var(1.2);
  prior.mutable_fixed_values()->set_shape(2.0);
  prior.mutable_fixed_values()->set_scale(2.0);

  // Initialize prior model
  auto prior_model = std::make_shared<NxIGPriorModel>();
  prior_model->get_mutable_prior()->CopyFrom(prior);
  prior_model->initialize();

  // Check equality before update
  prior_models.push_back(prior_model);
  for (size_t i = 1; i < 4; i++) {
    prior_models.push_back(prior_model->clone());
    prior_models[i]->write_hypers_to_proto(&prior_out);
    ASSERT_EQ(prior.fixed_values().DebugString(),
              prior_out.nnxig_state().DebugString());
  }

  // Check equality after update
  prior_models[0]->update_hypers(states);
  prior_models[0]->write_hypers_to_proto(&prior_out);
  for (size_t i = 1; i < 4; i++) {
    prior_models[i]->write_hypers_to_proto(&prior_out);
    ASSERT_EQ(prior.fixed_values().DebugString(),
              prior_out.nnxig_state().DebugString());
  }
}

TEST(nxig_prior_model, sample) {
  // Instance
  auto prior = std::make_shared<NxIGPriorModel>();
  bool use_post_hypers = true;

  // Define prior hypers
  bayesmix::AlgorithmState::HierarchyHypers hypers_proto;
  hypers_proto.mutable_nnxig_state()->set_mean(5.0);
  hypers_proto.mutable_nnxig_state()->set_var(1.2);
  hypers_proto.mutable_nnxig_state()->set_shape(4.0);
  hypers_proto.mutable_nnxig_state()->set_scale(3.0);

  // Set hypers and get sampled state as proto
  prior->set_hypers_from_proto(hypers_proto);
  auto state1 = prior->sample(!use_post_hypers);
  auto state2 = prior->sample(!use_post_hypers);

  // Check if they coincides
  ASSERT_TRUE(state1->DebugString() != state2->DebugString());
}