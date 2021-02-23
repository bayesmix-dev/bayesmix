#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "src/hierarchies/abstract_hierarchy.h"
#include "src/mixings/logit_sb_mixing.h"
#include "src/utils/io_utils.h"
#include "src/utils/proto_utils.h"
#include "src/utils/rng.h"

TEST(logit_sb, misc) {
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(20201124);

  LogitSBMixing mix;
  bayesmix::LogSBPrior prior;

  double step = 0.5;
  unsigned int num_clusters = 10;
  Eigen::Vector3d mu;
  mu << -2.0, 2.0, 0.0;
  auto cov = 2.0 * Eigen::MatrixXd::Identity(3, 3);

  prior.set_step_size(step);
  prior.set_num_clusters(num_clusters);
  bayesmix::to_proto(mu, prior.mutable_normal_prior()->mutable_mean());
  bayesmix::to_proto(cov, prior.mutable_normal_prior()->mutable_var());

  mix.get_mutable_prior()->CopyFrom(prior);

  std::string covsfile = "resources/test/mh_covs.csv";
  Eigen::MatrixXd covariates = bayesmix::read_eigen_matrix(covsfile);
  mix.set_covariates(&covariates);

  mix.initialize();

  google::protobuf::Message *prior_out = mix.get_mutable_prior();
  ASSERT_EQ(prior.DebugString(), prior_out->DebugString());

  Eigen::MatrixXd coeffs = mix.get_state().regression_coeffs;
  std::cout << coeffs << std::endl;
  for (int i = 0; i < coeffs.cols(); i++) {
    // Check equality of i-th column, element-by-element
    for (int j = 0; j < mu.size(); j++) {
      ASSERT_DOUBLE_EQ(mu(j), coeffs(j, i));
    }
  }

  unsigned int n_data = 200;
  std::vector<std::shared_ptr<AbstractHierarchy>> hierarchies(n_data);
  std::vector<unsigned int> allocations(n_data, 0);
  for (int i = n_data / 2; i < n_data; i++) {
    allocations[i] = 1;
  }

  unsigned int n_iter = 100;
  for (int i = 0; i < n_iter; i++) {
    mix.update_state(hierarchies, allocations);
  }

  Eigen::MatrixXd coeffs_out = mix.get_state().regression_coeffs;
  std::cout << coeffs_out << std::endl;
}
