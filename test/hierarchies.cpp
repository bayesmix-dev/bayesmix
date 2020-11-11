#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../proto/cpp/ls_state.pb.h"
#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"

TEST(nnighierarchy, draw) {
  auto hier = std::make_shared<NNIGHierarchy>();
  hier->set_mu0(5.0);
  hier->set_lambda0(0.1);
  hier->set_alpha0(2.0);
  hier->set_beta0(2.0);
  hier->check_and_initialize();

  auto hier2 = hier->clone();
  hier2->draw();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterVal* clusval = out.add_cluster_vals();
  bayesmix::MarginalState::ClusterVal* clusval2 = out.add_cluster_vals();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnighierarchy, sample_given_data) {
  auto hier = std::make_shared<NNIGHierarchy>();
  hier->set_mu0(5.0);
  hier->set_lambda0(0.1);
  hier->set_alpha0(2.0);
  hier->set_beta0(2.0);
  hier->check_and_initialize();

  Eigen::VectorXd datum(1);
  datum << 4.5;

  auto hier2 = hier->clone();
  hier2->sample_given_data(datum);

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterVal* clusval = out.add_cluster_vals();
  bayesmix::MarginalState::ClusterVal* clusval2 = out.add_cluster_vals();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnwhierarchy, draw) {
  auto hier = std::make_shared<NNWHierarchy>();
  Eigen::VectorXd mu0(2);
  mu0 << 5.5, 5.5;
  hier->set_mu0(mu0);
  hier->set_lambda0(0.2);
  double nu0 = 5.0;
  hier->set_nu0(nu0);
  Eigen::Matrix2d tau0 = (1 / nu0) * Eigen::Matrix2d::Identity();
  hier->set_tau0(tau0);
  hier->check_and_initialize();

  auto hier2 = hier->clone();
  hier2->draw();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterVal* clusval = out.add_cluster_vals();
  bayesmix::MarginalState::ClusterVal* clusval2 = out.add_cluster_vals();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnwhierarchy, sample_given_data) {
  auto hier = std::make_shared<NNWHierarchy>();
  Eigen::VectorXd mu0(2);
  mu0 << 5.5, 5.5;
  hier->set_mu0(mu0);
  hier->set_lambda0(0.2);
  double nu0 = 5.0;
  hier->set_nu0(nu0);
  Eigen::Matrix2d tau0 = (1 / nu0) * Eigen::Matrix2d::Identity();
  hier->set_tau0(tau0);
  hier->check_and_initialize();

  Eigen::RowVectorXd datum(2);
  datum << 4.5, 4.5;

  auto hier2 = hier->clone();
  hier2->sample_given_data(datum);

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterVal* clusval = out.add_cluster_vals();
  bayesmix::MarginalState::ClusterVal* clusval2 = out.add_cluster_vals();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}
