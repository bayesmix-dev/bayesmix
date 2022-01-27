#include "src/utils/proto_utils.h"

#include <gtest/gtest.h>

#include <stan/math/rev.hpp>

#include "matrix.pb.h"

TEST(to_proto, vector) {
  Eigen::VectorXd vec = Eigen::VectorXd::Ones(5);
  vec(1) = 100.0;

  bayesmix::Vector vecproto;
  bayesmix::to_proto(vec, &vecproto);

  ASSERT_EQ(vecproto.size(), 5);
  ASSERT_EQ(vecproto.data(0), 1.0);
  ASSERT_EQ(vecproto.data(1), 100.0);
}

TEST(to_proto, matrix) {
  Eigen::MatrixXd mat = Eigen::MatrixXd::Identity(5, 5);

  bayesmix::Matrix matproto;
  bayesmix::to_proto(mat, &matproto);

  ASSERT_EQ(matproto.rows(), 5);
  ASSERT_EQ(matproto.cols(), 5);
  ASSERT_EQ(matproto.data(0), 1.0);
  ASSERT_EQ(matproto.data(1), 0.0);

  mat(0, 1) = 100.0;
  bayesmix::to_proto(mat, &matproto);
  ASSERT_EQ(matproto.data(1), 0.0);
  ASSERT_EQ(matproto.data(5), 100.0);
}

TEST(to_eigen, vector) {
  Eigen::VectorXd vec = Eigen::VectorXd::Ones(5);
  vec(1) = 100.0;

  bayesmix::Vector vecproto;
  bayesmix::to_proto(vec, &vecproto);

  Eigen::VectorXd vecnew = bayesmix::to_eigen(vecproto);

  ASSERT_EQ(vecnew.size(), vec.size());
  ASSERT_EQ(vecnew.sum(), vec.sum());
}

TEST(to_eigen, matrix_colmajor) {
  Eigen::MatrixXd mat = Eigen::MatrixXd::Identity(5, 5);
  mat(0, 1) = 100.0;

  bayesmix::Matrix matproto;
  bayesmix::to_proto(mat, &matproto);

  Eigen::MatrixXd matnew = bayesmix::to_eigen(matproto);

  ASSERT_EQ(matnew.rows(), 5);
  ASSERT_EQ(matnew.cols(), 5);
  ASSERT_EQ(matnew(0, 0), 1.0);
  ASSERT_EQ(matnew(1, 0), 0.0);
  ASSERT_EQ(matnew(0, 1), 100.0);
}

TEST(to_eigen, matrix_rowmajor) {
  Eigen::MatrixXd mat = Eigen::MatrixXd::Identity(5, 5);
  mat(0, 1) = 100.0;

  bayesmix::Matrix matproto;
  bayesmix::to_proto(mat, &matproto);
  matproto.set_rowmajor(true);

  Eigen::MatrixXd matnew = bayesmix::to_eigen(matproto);

  ASSERT_EQ(matnew.rows(), 5);
  ASSERT_EQ(matnew.cols(), 5);
  ASSERT_EQ(matnew(0, 0), 1.0);
  ASSERT_FALSE(matnew(0, 1) == 100.0);
}
