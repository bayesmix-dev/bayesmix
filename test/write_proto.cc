#include <gtest/gtest.h>

#include "algorithm_state.pb.h"
#include "ls_state.pb.h"
#include "src/hierarchies/nnig_hierarchy.h"
#include "src/utils/proto_utils.h"

TEST(set_state, uni_ls) {
  double mean = 5;
  double var = 1.0;

  bayesmix::UniLSState curr;
  curr.set_mean(mean);
  curr.set_var(var);

  ASSERT_EQ(curr.mean(), mean);

  bayesmix::AlgorithmState::ClusterState clusval;
  clusval.mutable_uni_ls_state()->CopyFrom(curr);
  NNIGHierarchy cluster;
  cluster.set_state_from_proto(clusval);

  ASSERT_EQ(curr.mean(), cluster.get_state().mean);
}

TEST(write_proto, uni_ls) {
  double mean = 5;
  double var = 1.0;

  bayesmix::UniLSState curr;
  curr.set_mean(mean);
  curr.set_var(var);

  bayesmix::AlgorithmState::ClusterState clusval_in;
  clusval_in.mutable_uni_ls_state()->CopyFrom(curr);
  NNIGHierarchy cluster;
  cluster.set_state_from_proto(clusval_in);

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  cluster.write_state_to_proto(clusval);

  double out_mean = clusval->uni_ls_state().mean();
  double out_var = clusval->uni_ls_state().var();
  ASSERT_EQ(mean, out_mean);
  ASSERT_EQ(var, out_var);
}
