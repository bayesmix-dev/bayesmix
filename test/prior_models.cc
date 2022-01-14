#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim.hpp>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
// #include "ls_state.pb.h"
#include "src/hierarchies/priors/nig_prior_model.h"
// #include "src/utils/rng.h"

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

// TODO: test for the other priors available
TEST(nig_prior_model, fixed_values_prior) {
  bayesmix::NNIGPrior prior;
  bayesmix::AlgorithmState::HierarchyHypers prior_out;
  prior.mutable_fixed_values()->set_mean(5.0);
  prior.mutable_fixed_values()->set_var_scaling(0.1);
  prior.mutable_fixed_values()->set_shape(2.0);
  prior.mutable_fixed_values()->set_scale(2.0);

  auto prior_model = std::make_shared<NIGPriorModel>();
  prior_model->get_mutable_prior()->CopyFrom(prior);
  prior_model->initialize();

  std::vector<std::shared_ptr<AbstractPriorModel>> prior_models;
  std::vector<bayesmix::AlgorithmState::ClusterState> states;

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
