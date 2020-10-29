#include <gtest/gtest.h>
#include <iostream>

#include "../proto/cpp/ls_state.pb.h"
#include "../proto/cpp/marginal_state.pb.h"

#include "../src/hierarchies/HierarchyNNIG.hpp"


TEST(set_state, univ_ls) {
  using namespace bayesmix;
  double mean = 5;
  double std = 1.0;

  UnivLSState curr;
  curr.set_mean(mean);
  curr.set_std(std);

  ASSERT_EQ(curr.mean(), mean);

  HierarchyNNIG cluster;
  cluster.set_state(&curr);

  ASSERT_EQ(curr.mean(), cluster.get_mean());
}


TEST(get_state_as_proto_test, univ_ls) {
  using namespace bayesmix;
  double mean = 5;
  double std = 1.0;

  UnivLSState curr;
  curr.set_mean(mean);
  curr.set_std(std);

  HierarchyNNIG cluster;
  cluster.set_state(&curr);

  MarginalState out;
  MarginalState::ClusterVal* clusval = out.add_cluster_vals();

  cluster.get_state_as_proto(clusval);

  double out_mean = clusval->univ_ls_state().mean();
  double out_std = clusval->univ_ls_state().std();
  ASSERT_EQ(mean, out_mean);
  ASSERT_EQ(std, out_std);
}