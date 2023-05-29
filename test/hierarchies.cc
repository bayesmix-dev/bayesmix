#include <gtest/gtest.h>

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>

#include "algorithm_state.pb.h"
#include "ls_state.pb.h"
#include "src/hierarchies/betagg_hierarchy.h"
#include "src/hierarchies/fa_hierarchy.h"
#include "src/hierarchies/lin_reg_uni_hierarchy.h"
#include "src/hierarchies/nnig_hierarchy.h"
#include "src/hierarchies/nnw_hierarchy.h"
#include "src/hierarchies/nnxig_hierarchy.h"
#include "src/includes.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

TEST(nnig_hierarchy, draw) {
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

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnig_hierarchy, sample_given_data) {
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

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnw_hierarchy, draw) {
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

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnw_hierarchy, sample_given_data) {
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

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnw_hierarchy, no_unconstrained_lpdf) {
  // Initialize hierarchy
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

  // Check exeption handling in case unconstrained lpdfs are not implemented
  auto state_uc = hier->get_state().get_unconstrained();
  EXPECT_ANY_THROW(
      hier->get_likelihood()->cluster_lpdf_from_unconstrained(state_uc));
  EXPECT_ANY_THROW(hier->get_prior()->lpdf_from_unconstrained(state_uc));
}

TEST(lin_reg_uni_hierarchy, state_read_write) {
  Eigen::Vector2d beta;
  beta << 2, -1;
  double sigma2 = 9;

  bayesmix::LinRegUniLSState ls;
  bayesmix::to_proto(beta, ls.mutable_regression_coeffs());
  ls.set_var(sigma2);

  bayesmix::AlgorithmState::ClusterState state;
  state.mutable_lin_reg_uni_ls_state()->CopyFrom(ls);

  LinRegUniHierarchy hier;
  hier.set_state_from_proto(state);

  ASSERT_EQ(hier.get_state().regression_coeffs, beta);
  ASSERT_EQ(hier.get_state().var, sigma2);

  bayesmix::AlgorithmState outt;
  bayesmix::AlgorithmState::ClusterState* out = outt.add_cluster_states();
  hier.write_state_to_proto(out);
  ASSERT_EQ(beta, bayesmix::to_eigen(
                      out->lin_reg_uni_ls_state().regression_coeffs()));
  ASSERT_EQ(sigma2, out->lin_reg_uni_ls_state().var());
}

TEST(lin_reg_uni_hierarchy, misc) {
  // Build data
  int n = 5, dim = 2;
  Eigen::Vector2d beta_true({10.0, 10.0});
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
  auto Lambda0 = Eigen::Matrix2d::Identity();
  double a0 = 2.0;
  double b0 = 1.0;

  // Initialize hierarchy
  bayesmix::to_proto(beta0, prior.mutable_fixed_values()->mutable_mean());
  bayesmix::to_proto(Lambda0,
                     prior.mutable_fixed_values()->mutable_var_scaling());
  prior.mutable_fixed_values()->set_shape(a0);
  prior.mutable_fixed_values()->set_scale(b0);
  hier.get_mutable_prior()->CopyFrom(prior);
  hier.initialize();

  // Extract hypers for reading test
  bayesmix::AlgorithmState::HierarchyHypers out;
  hier.write_hypers_to_proto(&out);
  ASSERT_EQ(beta0, bayesmix::to_eigen(out.lin_reg_uni_state().mean()));
  ASSERT_EQ(Lambda0,
            bayesmix::to_eigen(out.lin_reg_uni_state().var_scaling()));
  ASSERT_EQ(a0, out.lin_reg_uni_state().shape());
  ASSERT_EQ(b0, out.lin_reg_uni_state().scale());

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

TEST(nnxig_hierarchy, draw) {
  auto hier = std::make_shared<NNxIGHierarchy>();
  bayesmix::NNxIGPrior prior;
  prior.mutable_fixed_values()->set_mean(5.0);
  prior.mutable_fixed_values()->set_var(1.2);
  prior.mutable_fixed_values()->set_shape(2.0);
  prior.mutable_fixed_values()->set_scale(2.0);
  hier->get_mutable_prior()->CopyFrom(prior);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->sample_prior();

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(nnxig_hierarchy, sample_given_data) {
  auto hier = std::make_shared<NNxIGHierarchy>();
  bayesmix::NNxIGPrior prior;
  prior.mutable_fixed_values()->set_mean(5.0);
  prior.mutable_fixed_values()->set_var(1.2);
  prior.mutable_fixed_values()->set_shape(2.0);
  prior.mutable_fixed_values()->set_scale(2.0);
  hier->get_mutable_prior()->CopyFrom(prior);

  hier->initialize();

  Eigen::VectorXd datum(1);
  datum << 4.5;

  auto hier2 = hier->clone();
  hier2->add_datum(0, datum, false);
  hier2->sample_full_cond();

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(fa_hierarchy, draw) {
  auto hier = std::make_shared<FAHierarchy>();
  bayesmix::FAPrior prior;
  Eigen::VectorXd mutilde(4);
  mutilde << 3.0, 3.0, 4.0, 1.0;
  bayesmix::Vector mutilde_proto;
  bayesmix::to_proto(mutilde, &mutilde_proto);
  int q = 2;
  double phi = 1.0;
  double alpha0 = 5.0;
  Eigen::VectorXd beta(4);
  beta << 3.0, 3.0, 2.0, 2.1;
  bayesmix::Vector beta_proto;
  bayesmix::to_proto(beta, &beta_proto);
  *prior.mutable_fixed_values()->mutable_mutilde() = mutilde_proto;
  prior.mutable_fixed_values()->set_phi(phi);
  prior.mutable_fixed_values()->set_alpha0(alpha0);
  prior.mutable_fixed_values()->set_q(q);
  *prior.mutable_fixed_values()->mutable_beta() = beta_proto;
  hier->get_mutable_prior()->CopyFrom(prior);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->sample_prior();

  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(fa_hierarchy, sample_given_data) {
  auto hier = std::make_shared<FAHierarchy>();
  bayesmix::FAPrior prior;
  Eigen::VectorXd mutilde(4);
  mutilde << 3.0, 3.0, 4.0, 1.0;
  bayesmix::Vector mutilde_proto;
  bayesmix::to_proto(mutilde, &mutilde_proto);
  int q = 2;
  double phi = 1.0;
  double alpha0 = 5.0;
  Eigen::VectorXd beta(4);
  beta << 3.0, 3.0, 2.0, 2.1;
  bayesmix::Vector beta_proto;
  bayesmix::to_proto(beta, &beta_proto);
  *prior.mutable_fixed_values()->mutable_mutilde() = mutilde_proto;
  prior.mutable_fixed_values()->set_phi(phi);
  prior.mutable_fixed_values()->set_alpha0(alpha0);
  prior.mutable_fixed_values()->set_q(q);
  *prior.mutable_fixed_values()->mutable_beta() = beta_proto;
  hier->get_mutable_prior()->CopyFrom(prior);
  Eigen::MatrixXd dataset(5, 4);
  dataset << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      20;
  hier->set_dataset(&dataset);
  hier->initialize();

  auto hier2 = hier->clone();
  hier2->add_datum(0, dataset.row(0), false);
  hier2->add_datum(1, dataset.row(1), false);
  hier2->sample_full_cond();
  bayesmix::AlgorithmState out;
  bayesmix::AlgorithmState::ClusterState* clusval = out.add_cluster_states();
  bayesmix::AlgorithmState::ClusterState* clusval2 = out.add_cluster_states();
  hier->write_state_to_proto(clusval);
  hier2->write_state_to_proto(clusval2);

  ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}

TEST(betagg_hierarchy, sample_given_data) {
  auto hier = std::make_shared<BetaGGHierarchy>();
  bayesmix::GGPrior prior;

  prior.mutable_fixed_values()->set_a_rate(2.0);
  prior.mutable_fixed_values()->set_a_shape(2.0);
  prior.mutable_fixed_values()->set_b_rate(2.0);
  prior.mutable_fixed_values()->set_b_shape(2.0);
  hier->get_mutable_prior()->CopyFrom(prior);

  auto mala_updater = std::make_shared<MalaUpdater>(0.0005);
  hier->set_updater(mala_updater);

  Eigen::MatrixXd dataset = Eigen::MatrixXd::Ones(100, 1) * 0.1;
  hier->set_dataset(&dataset);
  hier->initialize();

  for (int i = 0; i < dataset.rows(); i++) {
    hier->add_datum(i, dataset.row(i));
  }

  for (int i = 0; i < 1000; i++) {
    hier->sample_full_cond();
  }

  bayesmix::AlgorithmState::ClusterState out;
  hier->write_state_to_proto(&out);

  double a = out.sr_state().shape();
  double b = out.sr_state().rate();

  double mean = a / (a + b);
  double var = a * b / ((a + b) * (a + b) * (a + b + 1));

  ASSERT_LT(mean, 0.2);
  ASSERT_GT(mean, 0.05);
  ASSERT_LT(var, 0.1);

  // ASSERT_TRUE(clusval->DebugString() != clusval2->DebugString());
}
