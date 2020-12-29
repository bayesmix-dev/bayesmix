#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "../proto/cpp/semihdp.pb.h"
#include "../src/algorithms/semihdp_sampler.hpp"
#include "../src/includes.hpp"
#include "../src/utils/eigen_utils.hpp"

bayesmix::SemiHdpParams get_params() {
  bayesmix::SemiHdpParams out;
  out.mutable_pseudo_prior()->set_card_weight(0.5);
  out.mutable_pseudo_prior()->set_mean_perturb_sd(0.5);
  out.mutable_pseudo_prior()->set_var_perturb_frac(0.5);
  out.set_dirichlet_concentration(1.0);
  out.set_rest_allocs_update("full");
  out.set_totalmass_rest(1.0);
  out.set_totalmass_hdp(1.0);
  out.mutable_w_prior()->set_shape1(2.0);
  out.mutable_w_prior()->set_shape2(2.0);
  return out;
}

std::shared_ptr<BaseHierarchy> get_hierarchy() {
  auto hier = std::make_shared<NNIGHierarchy>();
  bayesmix::NNIGPrior hier_prior;
  hier_prior.mutable_fixed_values()->set_mean(0.0);
  hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
  hier_prior.mutable_fixed_values()->set_shape(2.0);
  hier_prior.mutable_fixed_values()->set_scale(2.0);
  hier->set_prior(hier_prior);
  return hier;
}

TEST(semihdp, relabel) {
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

  SemiHdpSampler sampler(data, get_hierarchy(), get_params());
  sampler.initialize();

  sampler.set_rest_tables_debug(theta_star_sizes);
  sampler.set_private_tables_debug(theta_tilde_sizes);
  sampler.set_shared_tables_debug(tau_size);

  sampler.set_table_allocs(s);
  sampler.set_to_shared(t);
  sampler.set_to_private(v);
  sampler.set_rest_allocs({0, 1});

  sampler.relabel();

  std::vector<std::vector<int>> snew = sampler.get_table_allocs();
  ASSERT_EQ(*std::max_element(snew[0].begin(), snew[0].end()),
            *std::max_element(s[0].begin(), s[0].end()) - 1);
  ASSERT_EQ(*std::max_element(snew[1].begin(), snew[1].end()),
            *std::max_element(s[1].begin(), s[1].end()) - 2);

  std::vector<std::vector<int>> tnew = sampler.get_to_shared();

  ASSERT_EQ(1, 1);
}

TEST(semihdp, relabel2) {
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

  SemiHdpSampler sampler(data, get_hierarchy(), get_params());
  sampler.initialize();

  sampler.set_rest_tables_debug(theta_star_sizes);
  sampler.set_private_tables_debug(theta_tilde_sizes);
  sampler.set_shared_tables_debug(tau_size);

  sampler.set_table_allocs(s);
  sampler.set_to_shared(t);
  sampler.set_to_private(v);
  sampler.set_rest_allocs({0, 1});

  sampler.relabel();

  std::vector<std::vector<int>> snew = sampler.get_table_allocs();
  std::vector<std::vector<int>> tnew = sampler.get_to_shared();
  std::vector<std::vector<int>> vnew = sampler.get_to_private();

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

TEST(semihdp, sample_unique_values) {
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
  t[0] = {-1, -1};
  t[1] = {-1, -1};

  std::vector<std::vector<int>> v(2);
  v[0] = {0, 1};
  v[1] = {0, 1};

  SemiHdpSampler sampler(data, get_hierarchy(), get_params());
  sampler.initialize();

  sampler.set_table_allocs(s);
  sampler.set_to_shared(t);
  sampler.set_to_private(v);
  sampler.update_unique_vals();

  ASSERT_DOUBLE_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(0, 0))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_private_table(0, 0))
          ->get_state()
          .mean);

  ASSERT_DOUBLE_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(0, 1))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_private_table(0, 1))
          ->get_state()
          .mean);

  ASSERT_DOUBLE_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 0))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_private_table(1, 0))
          ->get_state()
          .mean);

  ASSERT_GT(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(0, 0))
          ->get_state()
          .mean,
      0);
  ASSERT_LT(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 0))
          ->get_state()
          .mean,
      0);
  ASSERT_GT(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 1))
          ->get_state()
          .mean,
      0);
}

TEST(semihdp, sample_unique_values2) {
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

  SemiHdpSampler sampler(data, get_hierarchy(), get_params());
  sampler.initialize();

  sampler.set_table_allocs(s);
  sampler.set_to_shared(t);
  sampler.set_to_private(v);
  sampler.update_unique_vals();

  ASSERT_FLOAT_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(0, 0))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 1))
          ->get_state()
          .mean);

  ASSERT_FLOAT_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(0, 0))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_shared_table(0))
          ->get_state()
          .mean);

  ASSERT_FLOAT_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(0, 1))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_private_table(0, 0))
          ->get_state()
          .mean);

  ASSERT_FLOAT_EQ(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 0))
          ->get_state()
          .mean,
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_private_table(1, 0))
          ->get_state()
          .mean);

  ASSERT_GT(std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_shared_table(0))
                ->get_state()
                .mean,
            0);
  ASSERT_LT(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 0))
          ->get_state()
          .mean,
      0);
  ASSERT_GT(
      std::dynamic_pointer_cast<NNIGHierarchy>(sampler.get_table(1, 1))
          ->get_state()
          .mean,
      0);
}

TEST(semihdp, sample_allocations1) {
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

  SemiHdpSampler sampler(data, get_hierarchy(), get_params());
  sampler.initialize();

  sampler.set_table_allocs(s);
  sampler.set_to_shared(t);
  sampler.set_to_private(v);
  sampler.update_unique_vals();

  sampler.set_table_allocs(swrong);
  sampler.update_table_allocs();

  std::vector<std::vector<int>> snew = sampler.get_table_allocs();

  ASSERT_EQ(snew[0][1], 0);
  ASSERT_EQ(1, 1);
}
