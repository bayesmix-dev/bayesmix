#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>

#include "ls_state.pb.h"
#include "marginal_state.pb.h"
#include "src/hierarchies/lin_reg_uni_hierarchy.h"
#include "src/hierarchies/nnig_hierarchy.h"
#include "src/hierarchies/nnw_hierarchy.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

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
  hier->get_mutable_prior()->CopyFrom(prior);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->sample_prior();

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
  hier->get_mutable_prior()->CopyFrom(prior);

  hier->initialize();

  Eigen::VectorXd datum(1);
  datum << 4.5;

  auto hier2 = hier->clone();
  hier2->add_datum(0, datum, false);
  hier2->sample_full_cond();

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
  hier->get_mutable_prior()->CopyFrom(prior);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->sample_prior();

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
  hier->get_mutable_prior()->CopyFrom(prior);
  hier->initialize();

  Eigen::RowVectorXd datum(2);
  datum << 4.5, 4.5;

  auto hier2 = hier->clone();
  hier2->add_datum(0, datum, false);
  hier2->sample_full_cond();

  bayesmix::MarginalState out;
  bayesmix::MarginalState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::MarginalState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(lin_reg_uni_hierarchy, state_read_write) {
  Eigen::Vector2d beta;
  beta << 2, -1;
  double sigma2 = 9;

  bayesmix::LinRegUniLSState ls;
  bayesmix::to_proto(beta, ls.mutable_regression_coeffs());
  ls.set_var(sigma2);

  bayesmix::MarginalState::ClusterState state;
  state.mutable_lin_reg_uni_ls_state()->CopyFrom(ls);

  LinRegUniHierarchy hier;
  hier.set_state_from_proto(state);

  ASSERT_EQ(hier.get_state().regression_coeffs, beta);
  ASSERT_EQ(hier.get_state().var, sigma2);

  bayesmix::MarginalState outt;
  bayesmix::MarginalState::ClusterState* out = outt.add_cluster_states();
  hier.write_state_to_proto(out);
  ASSERT_EQ(beta, bayesmix::to_eigen(
                      out->lin_reg_uni_ls_state().regression_coeffs()));
  ASSERT_EQ(sigma2, out->lin_reg_uni_ls_state().var());
}

TEST(lin_reg_uni_hierarchy, misc) {
  // Build data
  int n = 5;
  int dim = 2;
  Eigen::Vector2d beta_true;
  beta_true << 10.0, 10.0;
  Eigen::MatrixXd cov = Eigen::MatrixXd::Random(n, dim);  // each in U[-1,1]
  double sigma2 = 1.0;
  Eigen::VectorXd data(n);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < n; i++) {
    data(i) = stan::math::normal_rng(cov.row(i).dot(beta_true), sigma2, rng);
  }
  // Initialize objects
  LinRegUniHierarchy hier;
  bayesmix::LinRegUniPrior prior;
  // Create prior parameters
  Eigen::Vector2d beta0 = 0 * beta_true;
  bayesmix::Vector beta0_proto;
  bayesmix::to_proto(beta0, &beta0_proto);
  auto Lambda0 = Eigen::Matrix2d::Identity();
  bayesmix::Matrix Lambda0_proto;
  bayesmix::to_proto(Lambda0, &Lambda0_proto);
  double a0 = 2.0;
  double b0 = 1.0;
  // Initialize hierarchy
  *prior.mutable_fixed_values()->mutable_mean() = beta0_proto;
  *prior.mutable_fixed_values()->mutable_var_scaling() = Lambda0_proto;
  prior.mutable_fixed_values()->set_shape(a0);
  prior.mutable_fixed_values()->set_scale(b0);
  hier.get_mutable_prior()->CopyFrom(prior);
  hier.initialize();
  // Extract hypers for reading test
  bayesmix::LinRegUniPrior out;
  hier.write_hypers_to_proto(&out);
  ASSERT_EQ(beta0, bayesmix::to_eigen(out.fixed_values().mean()));
  ASSERT_EQ(Lambda0, bayesmix::to_eigen(out.fixed_values().var_scaling()));
  ASSERT_EQ(a0, out.fixed_values().shape());
  ASSERT_EQ(b0, out.fixed_values().scale());
  // Add data
  for (int i = 0; i < n; i++) {
    hier.add_datum(i, data.row(i), false, cov.row(i));
  }
  // Check summary statistics
  // for (int i = 0; i < dim; i++) {
  //   for (int j = 0; j < dim; j++) {
  //     ASSERT_DOUBLE_EQ(hier.get_covar_sum_squares()(i, j),
  //                     (cov.transpose() * cov)(i, j));
  //   }
  //   ASSERT_DOUBLE_EQ(hier.get_mixed_prod()(i), (cov.transpose() * data)(i));
  // }
  // Compute and check posterior values
  hier.sample_full_cond();
  auto state = hier.get_state();
  for (int i = 0; i < dim; i++) {
    ASSERT_GT(state.regression_coeffs(i), beta0(i));
  }
}
