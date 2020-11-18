#include "proto_utils.hpp"

#include <Eigen/Dense>

#include "../../proto/cpp/matrix.pb.h"

void bayesmix::to_proto(const Eigen::MatrixXd &mat, bayesmix::Matrix *out) {
  out->set_rows(mat.rows());
  out->set_cols(mat.cols());
  out->set_rowmajor(false);
  *out->mutable_data() = {mat.data(), mat.data() + mat.size()};
}

void bayesmix::to_proto(const Eigen::VectorXd &vec, bayesmix::Vector *out) {
  out->set_size(vec.size());
  *out->mutable_data() = {vec.data(), vec.data() + vec.size()};
}

Eigen::VectorXd bayesmix::to_eigen(const bayesmix::Vector &vec) {
  int size = vec.size();
  Eigen::VectorXd out;
  if (size > 0) {
    const double *p = &(vec.data())[0];
    out = Eigen::Map<const Eigen::VectorXd>(p, size);
  }
  return out;
}

Eigen::MatrixXd bayesmix::to_eigen(const bayesmix::Matrix &mat) {
  int nrow = mat.rows();
  int ncol = mat.cols();
  Eigen::MatrixXd out;
  if (nrow > 0 & ncol > 0) {
    const double *p = &(mat.data())[0];
    if (mat.rowmajor()) {
      out = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor> >(
          p, nrow, ncol);
    } else {
      out = Eigen::Map<const Eigen::MatrixXd>(p, nrow, ncol);
    }
  }
  return out;
}
