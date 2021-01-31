#include "src/utils/distributions.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <vector>

#include "src/utils/rng.hpp"

TEST(mix_dist, 1) {
  auto& rng = bayesmix::Rng::Instance().get();

  int nclus = 5;
  Eigen::VectorXd weights1 =
      stan::math::dirichlet_rng(Eigen::VectorXd::Ones(nclus), rng);
  Eigen::VectorXd means1(nclus);
  Eigen::VectorXd sds1(nclus);

  for (int i = 0; i < nclus; i++) {
    means1(i) = stan::math::normal_rng(0, 2, rng);
    sds1(i) = stan::math::uniform_rng(0.1, 2.0, rng);
  }

  int nclus2 = 10;
  Eigen::VectorXd weights2 =
      stan::math::dirichlet_rng(Eigen::VectorXd::Ones(nclus2), rng);
  Eigen::VectorXd means2(nclus2);
  Eigen::VectorXd sds2(nclus2);

  for (int i = 0; i < nclus2; i++) {
    means2(i) = stan::math::normal_rng(0, 2, rng);
    sds2(i) = stan::math::uniform_rng(0.1, 2.0, rng);
  }

  double dist = bayesmix::gaussian_mixture_dist(means1, sds1, weights1, means2,
                                                sds2, weights2);

  ASSERT_GE(dist, 0.0);
}

TEST(mix_dist, 2) {
  int nclus = 5;
  auto& rng = bayesmix::Rng::Instance().get();

  Eigen::VectorXd weights1 =
      stan::math::dirichlet_rng(Eigen::VectorXd::Ones(nclus), rng);
  Eigen::VectorXd means1(nclus);
  Eigen::VectorXd sds1(nclus);

  for (int i = 0; i < nclus; i++) {
    means1(i) = stan::math::normal_rng(0, 2, rng);
    sds1(i) = stan::math::uniform_rng(0.1, 2.0, rng);
  }

  double dist_to_self = bayesmix::gaussian_mixture_dist(
      means1, sds1, weights1, means1, sds1, weights1);

  ASSERT_DOUBLE_EQ(dist_to_self, 0.0);
}
