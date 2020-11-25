#include "../src/utils/rng.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../src/utils/distributions.hpp"
#include "../src/hierarchies/nnig_hierarchy.hpp"

TEST(rng, test1) {
  int rounds = 1000;
  std::vector<double> p = {0.8, 0.1, 0.1};

  int dim = p.size();
  Eigen::Map<Eigen::VectorXd> probas(p.data(), dim);
  std::vector<int> counts(dim);
  for (int i = 0; i < rounds; i++) {
    auto& rng = bayesmix::Rng::Instance().get();
    int c = bayesmix::categorical_rng(probas, rng);
    counts[c]++;
  }
  std::cout << "[          ] ----> ";
  for (int i = 0; i < dim; i++) {
    std::cout << counts[i] << " ";
  }
  std::cout << std::endl;
}

TEST(rng, test2) {
  int dim = 50;
  std::vector<double> p(dim, 1.0);
  p[0] = 10.0;
  Eigen::Map<Eigen::VectorXd> probas(p.data(), dim);
  probas = probas / probas.sum();

  auto& rng = bayesmix::Rng::Instance().get();
  int c1 = bayesmix::categorical_rng(probas, rng);

  auto& rng2 = bayesmix::Rng::Instance().get();
  int c2 = bayesmix::categorical_rng(probas, rng2);

  std::cout << "[          ] ----> ";
  std::cout << c1 << " " << c2 << " ";

  if (true) {
    auto& rng = bayesmix::Rng::Instance().get();
    int c3 = bayesmix::categorical_rng(probas, rng);
    std::cout << c3 << " ";
  }
  if (true) {
    auto& rng = bayesmix::Rng::Instance().get();
    int c4 = bayesmix::categorical_rng(probas, rng);
    std::cout << c4 << " " << std::endl;
  }
}

TEST(rng, test3) {
  NNIGHierarchy hierarchy;
  bayesmix::NNIGPrior hier_prior;
  hier_prior.mutable_fixed_values()->set_mean(0.0);
  hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
  hier_prior.mutable_fixed_values()->set_shape(2.0);
  hier_prior.mutable_fixed_values()->set_scale(2.0);
  hierarchy.set_prior(hier_prior);

  hierarchy.draw();
  double m1 = hierarchy.get_mean();
  double s1 = hierarchy.get_sd();

  hierarchy.draw();
  double m2 = hierarchy.get_mean();
  double s2 = hierarchy.get_sd();

  ASSERT_NE(m1, m2);
  ASSERT_NE(s1, s2);

  NNIGHierarchy hierarchy2 = hierarchy;
  hierarchy2.draw();

  double m3 = hierarchy2.get_mean();
  double s3 = hierarchy2.get_sd();

  ASSERT_NE(m1, m3);
  ASSERT_NE(s1, s3);

  ASSERT_NE(m3, m2);
  ASSERT_NE(s3, s2);
}
