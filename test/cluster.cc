#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "src/clustering/lossfunction/LossFunction.hpp"
#include "src/clustering/lossfunction/BinderLoss.hpp"
#include "src/clustering/lossfunction/VariationInformation.hpp"

TEST(loss, binder_loss) {
  LossFunction binder(0);
  binder = new BinderLoss(1, 1);

  Eigen::VectorXi cluster1 << 1, 1, 1;
  Eigen::VectorXi cluster2 << 1, 1, 2;

  binder.setCluster(cluster1, cluster2);
  ASSERT_EQ(1, binder.loss());

  cluster2 << 1, 2, 3;
  ASSERT_EQ(3, binder.loss());
}

TEST(loss, vi_loss) {
  LossFunction vi(0);
  vi = new VariationInformation(1, 1);

  Eigen::VectorXi cluster1 << 1, 1, 1;
  Eigen::VectorXi cluster2 << 1, 1, 2;


  binder.setCluster(cluster1, cluster2);
  ASSERT_EQ(1, binder.loss());

  cluster2 << 1, 2, 3;
  ASSERT_EQ(3, binder.loss());
}