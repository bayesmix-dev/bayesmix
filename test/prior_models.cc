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
