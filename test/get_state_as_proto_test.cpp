#include <gtest/gtest.h>

#include "../proto/cpp/ls_state.pb.h"
#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/HierarchyNNIG.hpp"
#include "../src/hierarchies/HierarchyNNW.hpp"


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

TEST(set_state, multi_ls) {
  using namespace bayesmix;
  Eigen::VectorXd mean = Eigen::VectorXd::Ones(5);
  Eigen::MatrixXd prec = Eigen::MatrixXd::Identity(5, 5);
  prec(1, 1) = 10.0;

  MultiLSState curr;
  to_proto(mean, curr.mutable_mean());
  to_proto(prec, curr.mutable_precision());

  ASSERT_EQ(curr.mean().data(0), 1.0);
  ASSERT_EQ(curr.precision().data(0), 1.0);
  ASSERT_EQ(curr.precision().data(6), 10.0);

  HierarchyNNW cluster;
  cluster.set_state(&curr);

  ASSERT_EQ(curr.mean().data(0), cluster.get_mean()(0));
  ASSERT_EQ(curr.precision().data(0), cluster.get_tau()(0, 0));
  ASSERT_EQ(curr.precision().data(6), cluster.get_tau()(1, 1));
}
