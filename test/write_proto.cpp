#include <gtest/gtest.h>

#include "../proto/cpp/ls_state.pb.h"
#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"
#include "../src/utils/proto_utils.hpp"

TEST(set_state, univ_ls) {
  double mean = 5;
  double std = 1.0;

  bayesmix::UnivLSState curr;
  curr.set_mean(mean);
  curr.set_std(std);

  ASSERT_EQ(curr.mean(), mean);

  bayesmix::MarginalState::ClusterVal clusval;
  clusval.mutable_univ_ls_state()->CopyFrom(curr);
  NNIGHierarchy cluster;
  cluster.set_state(&clusval);

  ASSERT_EQ(curr.mean(), cluster.get_mean());
}

TEST(write_proto, univ_ls) {
  double mean = 5;
  double std = 1.0;

  bayesmix::UnivLSState curr;
  curr.set_mean(mean);
  curr.set_std(std);

  bayesmix::MarginalState::ClusterVal clusval_in;
  clusval_in.mutable_univ_ls_state()->CopyFrom(curr);
  NNIGHierarchy cluster;
  cluster.set_state(&clusval_in);

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterVal* clusval = out.add_cluster_vals();
  cluster.write_state_to_proto(clusval);

  double out_mean = clusval->univ_ls_state().mean();
  double out_std = clusval->univ_ls_state().std();
  ASSERT_EQ(mean, out_mean);
  ASSERT_EQ(std, out_std);
}

TEST(set_state, multi_ls) {
  Eigen::VectorXd mean = Eigen::VectorXd::Ones(5);
  Eigen::MatrixXd prec = Eigen::MatrixXd::Identity(5, 5);
  prec(1, 1) = 10.0;

  bayesmix::MultiLSState curr;
  bayesmix::to_proto(mean, curr.mutable_mean());
  bayesmix::to_proto(prec, curr.mutable_precision());

  ASSERT_EQ(curr.mean().data(0), 1.0);
  ASSERT_EQ(curr.precision().data(0), 1.0);
  ASSERT_EQ(curr.precision().data(6), 10.0);

  bayesmix::MarginalState::ClusterVal clusval_in;
  clusval_in.mutable_multi_ls_state()->CopyFrom(curr);
  NNWHierarchy cluster;
  cluster.set_state(&clusval_in);

  ASSERT_EQ(curr.mean().data(0), cluster.get_mean()(0));
  ASSERT_EQ(curr.precision().data(0), cluster.get_tau()(0, 0));
  ASSERT_EQ(curr.precision().data(6), cluster.get_tau()(1, 1));
}
