#include <gtest/gtest.h>

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
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
  unsigned int n_comp = 3;
  unsigned int n_data = 90;
  unsigned int n_iter = 20;

  // DATA GENERATION
  // True coefficients
  Eigen::MatrixXd cov_centers(dim, n_comp);
  cov_centers.col(0) << -5.0, 0.0;
  cov_centers.col(1) << 5.0, 0.0;
  cov_centers.col(2) << 0.0, 5.0;
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
        stan::math::multi_normal_rng(cov_centers.col(allocations[i]),
                                     Eigen::MatrixXd::Identity(dim, dim), rng);
  }
  // INITIALIZATION
  // Prior parameters
  double step = 0.025;
  auto prior_mean = Eigen::VectorXd::Zero(dim);
  auto cov = 5.0 * Eigen::MatrixXd::Identity(dim, dim);
  // Set parameters to mixing object
  LogitSBMixing mix;
  bayesmix::LogSBPrior prior;
  prior.set_step_size(step);
  prior.set_num_components(n_comp);
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
  // M-H run
  std::vector<std::shared_ptr<AbstractHierarchy>> hierarchies(n_comp);
  for (int i = 0; i < n_iter; i++) {
    mix.update_state(hierarchies, allocations);
    // std::cout << i << "\n" << mix.get_state().regression_coeffs <<
    // std::endl;
  }
  std::cout << "acceptance rates: " << mix.get_acceptance_rates().transpose()
            << std::endl;
  // Weights with test set
  Eigen::VectorXd test1(2);
  test1 << -5, 0;
  Eigen::VectorXd test2(2);
  test2 << 5, 0;
  Eigen::VectorXd test3(2);
  test3 << 0, 5;
  std::cout << "test1: "
            << mix.get_mixing_weights(false, false, test1).transpose()
            << std::endl;
  std::cout << "test2: "
            << mix.get_mixing_weights(false, false, test2).transpose()
            << std::endl;
  std::cout << "test3: "
            << mix.get_mixing_weights(false, false, test3).transpose()
            << std::endl;
}
