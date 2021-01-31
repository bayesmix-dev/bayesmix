#include "../src/utils/eigen_utils.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

TEST(vstack, 1) {
  std::vector<Eigen::MatrixXd> mats(3);
  mats[0] = Eigen::MatrixXd::Ones(3, 2);
  mats[1] = Eigen::MatrixXd::Zero(1, 2);
  mats[2] = Eigen::MatrixXd(1, 2);
  mats[2] << 1, 2;

  EXPECT_NO_THROW(bayesmix::vstack(mats));
  Eigen::MatrixXd out = bayesmix::vstack(mats);

  ASSERT_EQ(out(0, 0), mats[0](0, 0));
  ASSERT_EQ(out(3, 0), mats[1](0, 0));
  ASSERT_EQ(out(4, 0), mats[2](0, 0));
  ASSERT_EQ(out(4, 1), mats[2](0, 1));

  mats[2] = Eigen::MatrixXd::Zero(1, 1);
  EXPECT_THROW(bayesmix::vstack(mats), std::invalid_argument);
}

TEST(append_by_row, inplace) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Ones(3, 2);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(1, 2);
  int init_a_rows = a.rows();

  EXPECT_NO_THROW(bayesmix::append_by_row(&a, b));
  ASSERT_EQ(a.rows(), init_a_rows + b.rows());
  ASSERT_EQ(a.cols(), 2);
  ASSERT_EQ(a(init_a_rows, 0), b(0, 0));
  ASSERT_EQ(a(init_a_rows, 1), b(0, 1));

  Eigen::MatrixXd a2 = Eigen::MatrixXd::Ones(0, 0);
  EXPECT_NO_THROW(bayesmix::append_by_row(&a2, b));
  ASSERT_EQ(a2.rows(), b.rows());
  ASSERT_EQ(a2(0, 0), b(0, 0));

  init_a_rows = a.rows();
  Eigen::MatrixXd b2 = Eigen::MatrixXd::Ones(0, 0);
  EXPECT_NO_THROW(bayesmix::append_by_row(&a, b2));
  ASSERT_EQ(a.rows(), init_a_rows);

  Eigen::MatrixXd b3 = Eigen::MatrixXd::Zero(1, 1);
  EXPECT_THROW(bayesmix::append_by_row(&a, b3), std::invalid_argument);
}

TEST(append_by_row, copy) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Ones(3, 2);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(1, 2);
  int init_a_rows = a.rows();

  EXPECT_NO_THROW(bayesmix::append_by_row(a, b));
  Eigen::MatrixXd anew = bayesmix::append_by_row(a, b);
  ASSERT_GT(anew.rows(), init_a_rows);
  ASSERT_EQ(anew.cols(), 2);
  ASSERT_EQ(anew(init_a_rows, 0), b(0, 0));
  ASSERT_EQ(anew(init_a_rows, 1), b(0, 1));

  Eigen::MatrixXd a2 = Eigen::MatrixXd::Ones(0, 0);
  EXPECT_NO_THROW(bayesmix::append_by_row(a2, b));
  Eigen::MatrixXd a2new = bayesmix::append_by_row(a2, b);
  ASSERT_EQ(a2new.rows(), b.rows());
  ASSERT_EQ(a2new(0, 0), b(0, 0));

  init_a_rows = a.rows();
  Eigen::MatrixXd b2 = Eigen::MatrixXd::Ones(0, 0);
  EXPECT_NO_THROW(bayesmix::append_by_row(a, b2));
  ASSERT_EQ(a.rows(), init_a_rows);

  Eigen::MatrixXd b3 = Eigen::MatrixXd::Zero(1, 1);
  EXPECT_THROW(bayesmix::append_by_row(a, b3), std::invalid_argument);
}
