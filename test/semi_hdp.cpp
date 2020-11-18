#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "../src/algorithms/semihdp_sampler.hpp"
#include "../src/utils/eigen_utils.hpp"

TEST(relabel, 1) {
  std::vector<Eigen::MatrixXd> data(2);
  data[0] = Eigen::MatrixXd::Zero(10, 1);
  data[1] = Eigen::MatrixXd::Zero(10, 1);

  std::vector<std::vector<int>> s(2);
  s[0] = {0, 1, 2, 2, 4, 5, 5, 5, 5, 5};
  s[1] = {0, 1, 1, 1, 4, 5, 5, 5, 5, 5};

  std::vector<std::vector<int>> t(2);
  t[0] = {0, -1, 1, -1, 2, -1, -1, -1, -1, -1};
  t[1] = {-1, -1, 3, -1, 2, -1, -1, -1, -1, -1};

  std::vector<std::vector<int>> v(2);
  v[0] = {-1, 0, -1, 1, -1, 2, -1, -1, -1, -1};
  v[1] = {0, 0, -1, 1, -1, 2, -1, -1, -1, -1};

  std::vector<int> theta_star_sizes{10, 10};
  std::vector<int> theta_tilde_sizes{4, 4};
  int tau_size = 10;

  SemiHdpSampler sampler(data);

  sampler.set_theta_star_debug(theta_star_sizes);
  sampler.set_theta_tilde_debug(theta_tilde_sizes);
  sampler.set_tau_debug(tau_size);

  sampler.set_s(s);
  sampler.set_t(t);
  sampler.set_v(v);
  sampler.set_c({0, 1});

  sampler.relabel();

  std::vector<std::vector<int>> snew = sampler.get_s();
  ASSERT_EQ(*std::max_element(snew[0].begin(), snew[0].end()),
            *std::max_element(s[0].begin(), s[0].end()) - 1);
  ASSERT_EQ(*std::max_element(snew[1].begin(), snew[1].end()),
            *std::max_element(s[1].begin(), s[1].end()) - 2);

  std::vector<std::vector<int>> tnew = sampler.get_t();

  ASSERT_EQ(1, 1);
}

TEST(relabel, 2) {
  std::vector<Eigen::MatrixXd> data(2);
  data[0] = Eigen::MatrixXd::Zero(10, 1);
  data[1] = Eigen::MatrixXd::Zero(10, 1);

  std::vector<std::vector<int>> s(2);
  s[0] = {0, 1, 2, 3, 4, 5, 5, 5, 5, 5};
  s[1] = {1, 1, 1, 1, 4, 5, 5, 5, 5, 5};

  std::vector<std::vector<int>> t(2);
  t[0] = {0, -1, 1, -1, 2, -1, -1, -1, -1, -1};
  t[1] = {-1, -1, 3, -1, 2, -1, -1, -1, -1, -1};

  std::vector<std::vector<int>> v(2);
  v[0] = {-1, 0, -1, 1, -1, 2, -1, -1, -1, -1};
  v[1] = {0, 1, -1, 2, -1, 3, -1, -1, -1, -1};

  std::vector<int> theta_star_sizes{10, 10};
  std::vector<int> theta_tilde_sizes{4, 4};
  int tau_size = 10;

  SemiHdpSampler sampler(data);

  sampler.set_theta_star_debug(theta_star_sizes);
  sampler.set_theta_tilde_debug(theta_tilde_sizes);
  sampler.set_tau_debug(tau_size);

  sampler.set_s(s);
  sampler.set_t(t);
  sampler.set_v(v);
  sampler.set_c({0, 1});

  sampler.relabel();

  std::vector<std::vector<int>> snew = sampler.get_s();
  std::vector<std::vector<int>> tnew = sampler.get_t();
  std::vector<std::vector<int>> vnew = sampler.get_v();

  ASSERT_EQ(*std::max_element(snew[0].begin(), snew[0].end()), 5);
  ASSERT_EQ(*std::max_element(snew[1].begin(), snew[1].end()), 2);

  for (int i = 0; i < 2; i++) {
    std::vector<int> v_sorted(vnew[i]);
    std::sort(v_sorted.begin(), v_sorted.end());

    auto it = std::upper_bound(v_sorted.begin(), v_sorted.end(), -1);
    if (it != v[i].end()) {
      int min_v = *it;
      ASSERT_EQ(min_v, 0);
    }
  }

  ASSERT_EQ(*std::max_element(vnew[0].begin(), vnew[0].end()),
            *std::max_element(v[0].begin(), v[0].end()));
  ASSERT_EQ(*std::max_element(vnew[1].begin(), vnew[1].end()),
            *std::max_element(v[1].begin(), v[1].end()) - 2);

  ASSERT_EQ(*std::max_element(tnew[1].begin(), tnew[1].end()), 2);

  ASSERT_EQ(1, 1);
}

TEST(sample_unique_values, 1) {
  std::vector<Eigen::MatrixXd> data(2);

  data[0] = bayesmix::vstack({Eigen::MatrixXd::Zero(50, 1).array() + 5,
                              Eigen::MatrixXd::Zero(50, 1).array()});

  data[1] = bayesmix::vstack({Eigen::MatrixXd::Zero(50, 1).array() - 5,
                              Eigen::MatrixXd::Zero(50, 1).array() + 5});

  std::vector<std::vector<int>> s(2);
  s[0].resize(100);
  s[1].resize(100);
  
  for (int i=0; i < 50; i++) {
      s[0][i] = 0;
      s[1][i] = 0;
      s[0][i + 50] = 1;
      s[1][i + 50] = 1;
  }

  std::vector<std::vector<int>> t(2);
  t[0] = {-1, -1};
  t[1] = {-1, -1};

  std::vector<std::vector<int>> v(2);
  v[0] = {0, 1};
  v[1] = {0, 1};


  SemiHdpSampler sampler(data);
  sampler.initialize();

  sampler.set_s(s);
  sampler.set_t(t);
  sampler.set_v(v);
  sampler.update_unique_vals();

  ASSERT_FLOAT_EQ(sampler.get_theta_star(0, 0).get_mean(),
                  sampler.get_theta_tilde(0, 0).get_mean());

  ASSERT_FLOAT_EQ(sampler.get_theta_star(0, 1).get_mean(),
                  sampler.get_theta_tilde(0, 1).get_mean());

  ASSERT_FLOAT_EQ(sampler.get_theta_star(1, 0).get_mean(),
                  sampler.get_theta_tilde(1, 0).get_mean());

  ASSERT_GT(sampler.get_theta_star(0, 0).get_mean(), 0);
  ASSERT_LT(sampler.get_theta_star(1, 0).get_mean(), 0);
  ASSERT_GT(sampler.get_theta_star(1, 1).get_mean(), 0);
}

TEST(sample_unique_values, 2) {
  std::vector<Eigen::MatrixXd> data(2);

  data[0] = bayesmix::vstack({Eigen::MatrixXd::Zero(50, 1).array() + 5,
                              Eigen::MatrixXd::Zero(50, 1).array()});

  data[1] = bayesmix::vstack({Eigen::MatrixXd::Zero(50, 1).array() - 5,
                              Eigen::MatrixXd::Zero(50, 1).array() + 5});

  std::vector<std::vector<int>> s(2);
  s[0].resize(100);
  s[1].resize(100);

  for (int i = 0; i < 50; i++) {
    s[0][i] = 0;
    s[1][i] = 0;
    s[0][i + 50] = 1;
    s[1][i + 50] = 1;
  }

  std::vector<std::vector<int>> t(2);
  t[0] = {0, -1};
  t[1] = {-1, 0};

  std::vector<std::vector<int>> v(2);
  v[0] = {-1, 0};
  v[1] = {0, -1};

  SemiHdpSampler sampler(data);
  sampler.initialize();

  sampler.set_s(s);
  sampler.set_t(t);
  sampler.set_v(v);
  sampler.update_unique_vals();

  ASSERT_FLOAT_EQ(sampler.get_theta_star(0, 0).get_mean(),
                  sampler.get_theta_star(1, 1).get_mean());

  ASSERT_FLOAT_EQ(sampler.get_theta_star(0, 0).get_mean(),
                  sampler.get_tau(0).get_mean());

  ASSERT_FLOAT_EQ(sampler.get_theta_star(0, 1).get_mean(),
                  sampler.get_theta_tilde(0, 0).get_mean());

  ASSERT_FLOAT_EQ(sampler.get_theta_star(1, 0).get_mean(),
                  sampler.get_theta_tilde(1, 0).get_mean());

  ASSERT_GT(sampler.get_tau(0).get_mean(), 0);
  ASSERT_LT(sampler.get_theta_star(1, 0).get_mean(), 0);
  ASSERT_GT(sampler.get_theta_star(1, 1).get_mean(), 0);

}

TEST(sample_allocations, 1) {
  std::vector<Eigen::MatrixXd> data(2);

  data[0] = bayesmix::vstack({Eigen::MatrixXd::Zero(50, 1).array() + 5,
                              Eigen::MatrixXd::Zero(50, 1).array()});

  data[1] = bayesmix::vstack({Eigen::MatrixXd::Zero(50, 1).array() - 5,
                              Eigen::MatrixXd::Zero(50, 1).array() + 5});

  std::vector<std::vector<int>> s(2);
  std::vector<std::vector<int>> swrong(2);
  s[0].resize(100);
  s[1].resize(100);
  swrong[0].resize(100);
  swrong[1].resize(100);

  for (int i = 0; i < 50; i++) {
    s[0][i] = 0;
    s[1][i] = 0;
    swrong[0][i] = 1;
    swrong[1][i] = 1;
    s[0][i + 50] = 1;
    s[1][i + 50] = 1;
    swrong[0][i + 50] = 0;
    swrong[1][i + 50] = 0;
  }

  std::vector<std::vector<int>> t(2);
  t[0] = {0, -1};
  t[1] = {-1, 0};

  std::vector<std::vector<int>> v(2);
  v[0] = {-1, 0};
  v[1] = {0, -1};

  SemiHdpSampler sampler(data);
  sampler.initialize();

  sampler.set_s(s);
  sampler.set_t(t);
  sampler.set_v(v);
  sampler.update_unique_vals();
  sampler.relabel();
  
  sampler.set_s(swrong);
  sampler.update_s();

  std::vector<std::vector<int>> snew = sampler.get_s();
   

  ASSERT_EQ(snew[0][1], 0);
  ASSERT_EQ(1, 1);
}