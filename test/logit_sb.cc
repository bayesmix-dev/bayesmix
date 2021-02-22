#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "src/mixings/logit_sb_mixing.h"
#include "src/utils/rng.h"
#include "src/utils/io_utils.h"
#include "src/utils/proto_utils.h"

TEST(logit_sb, read_write) {
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
  mix.initialize();

  // double m_mix = mix.get_state().totalmass;
  // ASSERT_DOUBLE_EQ(m_prior, m_mix);

  // std::vector<std::shared_ptr<AbstractHierarchy>> hiers(100);
  // unsigned int n_data = 1000;
  // mix.update_state(hiers, std::vector<unsigned int>(), n_data);
  // double m_mix_after = mix.get_state().totalmass;

  // std::cout << "             after = " << m_mix_after << std::endl;
  // ASSERT_TRUE(m_mix_after > m_mix);
}

TEST(logit_sb, metroplois) {
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(20201124);

  std::string datafile = "resources/test/mh_data.csv";
  std::string covsfile = "resources/test/mh_covs.csv";

  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd covariates = bayesmix::read_eigen_matrix(covsfile);
}
