#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../proto/cpp/ls_state.pb.h"
#include "../proto/cpp/marginal_state.pb.h"
#include "../src/hierarchies/nnig_hierarchy.hpp"
#include "../src/hierarchies/nnw_hierarchy.hpp"
#include "../src/utils/proto_utils.hpp"

TEST(nnighierarchy, draw) {
  auto hier = std::make_shared<NNIGHierarchy>();
  bayesmix::NNIGPrior prior;
  double mu0 = 5.0;
  double lambda0 = 0.1;
  double alpha0 = 2.0;
  double beta0 = 2.0;
  prior.mutable_fixed_values()->set_mean(mu0);
  prior.mutable_fixed_values()->set_var_scaling(lambda0);
  prior.mutable_fixed_values()->set_shape(alpha0);
  prior.mutable_fixed_values()->set_scale(beta0);
  hier->set_prior(prior);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->draw();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::MarginalState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnighierarchy, sample_given_data) {
  auto hier = std::make_shared<NNIGHierarchy>();
  bayesmix::NNIGPrior prior;
  double mu0 = 5.0;
  double lambda0 = 0.1;
  double alpha0 = 2.0;
  double beta0 = 2.0;
  prior.mutable_fixed_values()->set_mean(mu0);
  prior.mutable_fixed_values()->set_var_scaling(lambda0);
  prior.mutable_fixed_values()->set_shape(alpha0);
  prior.mutable_fixed_values()->set_scale(beta0);
  hier->set_prior(prior);
  hier->initialize();

  Eigen::VectorXd datum(1);
  datum << 4.5;

  auto hier2 = hier->clone();
  hier2->add_datum(0, datum);
  hier2->sample_given_data();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::MarginalState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnwhierarchy, draw) {
  auto hier = std::make_shared<NNWHierarchy>();
  bayesmix::NNWPrior prior;
  Eigen::Vector2d mu0;
  mu0 << 5.5, 5.5;
  bayesmix::Vector mu0_proto;
  bayesmix::to_proto(mu0, &mu0_proto);
  double lambda0 = 0.2;
  double nu0 = 5.0;
  Eigen::Matrix2d tau0 = Eigen::Matrix2d::Identity() / nu0;
  bayesmix::Matrix tau0_proto;
  bayesmix::to_proto(tau0, &tau0_proto);
  *prior.mutable_fixed_values()->mutable_mean() = mu0_proto;
  prior.mutable_fixed_values()->set_var_scaling(lambda0);
  prior.mutable_fixed_values()->set_deg_free(nu0);
  *prior.mutable_fixed_values()->mutable_scale() = tau0_proto;
  hier->set_prior(prior);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->draw();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::MarginalState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnwhierarchy, sample_given_data) {
  auto hier = std::make_shared<NNWHierarchy>();
  bayesmix::NNWPrior prior;
  Eigen::Vector2d mu0;
  mu0 << 5.5, 5.5;
  bayesmix::Vector mu0_proto;
  bayesmix::to_proto(mu0, &mu0_proto);
  double lambda0 = 0.2;
  double nu0 = 5.0;
  Eigen::Matrix2d tau0 = Eigen::Matrix2d::Identity() / nu0;
  bayesmix::Matrix tau0_proto;
  bayesmix::to_proto(tau0, &tau0_proto);
  *prior.mutable_fixed_values()->mutable_mean() = mu0_proto;
  prior.mutable_fixed_values()->set_var_scaling(lambda0);
  prior.mutable_fixed_values()->set_deg_free(nu0);
  *prior.mutable_fixed_values()->mutable_scale() = tau0_proto;
  hier->set_prior(prior);
  hier->initialize();

  Eigen::RowVectorXd datum(2);
  datum << 4.5, 4.5;

  auto hier2 = hier->clone();
  hier2 -> add_datum(0, datum);
  hier2->sample_given_data();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::MarginalState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}
