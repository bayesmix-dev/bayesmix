#include <gtest/gtest.h>

#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"
#include "../src/utils/proto_utils.hpp"
#include "ls_state.pb.h"
#include "marginal_state.pb.h"

TEST(set_state, univ_ls) {
  double mean = 5;
  double var = 1.0;

  bayesmix::UnivLSState curr;
  curr.set_mean(mean);
  curr.set_var(var);

  ASSERT_EQ(curr.mean(), mean);

  bayesmix::MarginalState::ClusterState clusval;
  clusval.mutable_univ_ls_state()->CopyFrom(curr);
  NNIGHierarchy cluster;
  cluster.set_state_from_proto(clusval);

  ASSERT_EQ(curr.mean(), cluster.get_state().mean);
}

TEST(write_proto, univ_ls) {
  double mean = 5;
  double var = 1.0;

  bayesmix::UnivLSState curr;
  curr.set_mean(mean);
  curr.set_var(var);

  bayesmix::MarginalState::ClusterState clusval_in;
  clusval_in.mutable_univ_ls_state()->CopyFrom(curr);
  NNIGHierarchy cluster;
  cluster.set_state_from_proto(clusval_in);

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterState* clusval = out.add_cluster_states();
  cluster.write_state_to_proto(clusval);

  double out_mean = clusval->univ_ls_state().mean();
  double out_var = clusval->univ_ls_state().var();
  ASSERT_EQ(mean, out_mean);
  ASSERT_EQ(var, out_var);
}

TEST(set_state, multi_ls) {
  Eigen::VectorXd mean = Eigen::VectorXd::Ones(5);
  Eigen::MatrixXd prec = Eigen::MatrixXd::Identity(5, 5);
  prec(1, 1) = 10.0;

  bayesmix::MultiLSState curr;
  bayesmix::to_proto(mean, curr.mutable_mean());
  bayesmix::to_proto(prec, curr.mutable_prec());

  ASSERT_EQ(curr.mean().data(0), 1.0);
  ASSERT_EQ(curr.prec().data(0), 1.0);
  ASSERT_EQ(curr.prec().data(6), 10.0);

  bayesmix::MarginalState::ClusterState clusval_in;
  clusval_in.mutable_multi_ls_state()->CopyFrom(curr);
  NNWHierarchy cluster;
  cluster.set_state_from_proto(clusval_in);

  ASSERT_EQ(curr.mean().data(0), cluster.get_state().mean(0));
  ASSERT_EQ(curr.prec().data(0), cluster.get_state().prec(0, 0));
  ASSERT_EQ(curr.prec().data(6), cluster.get_state().prec(1, 1));
}
