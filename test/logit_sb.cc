#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

#include "src/hierarchies/abstract_hierarchy.h"
#include "src/mixings/logit_sb_mixing.h"
#include "src/utils/io_utils.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

TEST(logit_sb, misc) {
  // Initialize parameters
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(20201124);
  unsigned int dim = 2;
  unsigned int n_clust = 3;
  unsigned int n_data = 90;
  unsigned int n_iter = 200;

  // DATA GENERATION
  // True coefficients
  Eigen::MatrixXd true_alphas(dim, n_clust);
  for (int i = 0; i < n_clust; i++) {
    for (int j = 0; j < dim; j++) {
      true_alphas(j, i) = std::pow(2.0, i + 1);
    }
  }
  // Allocations
  std::vector<unsigned int> allocations(n_data);
  for (int i = 0; i < n_data / 3; i++) {
    allocations[n_data / 3 + i] = 1;
    allocations[2 * n_data / 3 + i] = 2;
  }
  // Covariates, based on cluster allocation
  Eigen::MatrixXd covariates(n_data, dim);
  for (int i = 0; i < n_data; i++) {
    covariates.row(i) =
        stan::math::multi_normal_rng(true_alphas.col(allocations[i]),
                                     Eigen::MatrixXd::Identity(dim, dim), rng);
  }

  // INITIALIZATION
  // Prior parameters
  double step = 0.5;
  auto prior_mean = Eigen::VectorXd::Zero(dim);
  auto cov = 2.0 * Eigen::MatrixXd::Identity(dim, dim);
  // Set parameters to mixing object
  LogitSBMixing mix;
  bayesmix::LogSBPrior prior;
  prior.set_step_size(step);
  prior.set_num_clusters(n_clust);
  bayesmix::to_proto(prior_mean, prior.mutable_normal_prior()->mutable_mean());
  bayesmix::to_proto(cov, prior.mutable_normal_prior()->mutable_var());
  mix.get_mutable_prior()->CopyFrom(prior);
  mix.set_covariates(&covariates);
  mix.initialize();

  // TESTING
  // Test prior read/write
  google::protobuf::Message *prior_out = mix.get_mutable_prior();
  ASSERT_EQ(prior.DebugString(), prior_out->DebugString());
  // Test initialization of state
  Eigen::MatrixXd coeffs = mix.get_state().regression_coeffs;
  for (int i = 0; i < coeffs.cols(); i++) {
    // Check equality of i-th column, element-by-element
    for (int j = 0; j < prior_mean.size(); j++) {
      ASSERT_DOUBLE_EQ(prior_mean(j), coeffs(j, i));
    }
  }
  // Statistical test
  std::vector<std::shared_ptr<AbstractHierarchy>> hierarchies(n_clust);
  for (int i = 0; i < n_iter; i++) {
    mix.update_state(hierarchies, allocations);
    std::cout << i << "\n" << mix.get_state().regression_coeffs << std::endl;
  }
}
