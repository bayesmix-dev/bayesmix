#include "proto_utils.hpp"

//! \param un_val Unique value in Protobuf-object form
//! \return       Matrix version of un_val
Eigen::MatrixXd bayesmix::proto_param_to_matrix(const Param &un_val) {
  Eigen::MatrixXd par_matrix = Eigen::MatrixXd::Zero(
      un_val.par_cols(0).elems_size(), un_val.par_cols_size());

  // Loop over unique values to copy them one at a time
  for (size_t i = 0; i < un_val.par_cols_size(); i++) {
    for (size_t j = 0; j < un_val.par_cols(i).elems_size(); j++) {
      par_matrix(j, i) = un_val.par_cols(i).elems(j);
    }
  }
  return par_matrix;
}

void bayesmix::to_proto(const Eigen::MatrixXd &mat, Matrix *out) {
  out->set_rows(mat.rows());
  out->set_cols(mat.cols());
  *out->mutable_data() = {mat.data(), mat.data() + mat.size()};
}

void bayesmix::to_proto(const Eigen::VectorXd &vec, Vector *out) {
  out->set_size(vec.size());
  *out->mutable_data() = {vec.data(), vec.data() + vec.size()};
}

Eigen::VectorXd bayesmix::to_eigen(const Vector &vec) {
  int size = vec.size();
  Eigen::VectorXd out;
  if (size > 0) {
    const double *p = &(vec.data())[0];
    out = Eigen::Map<const Eigen::VectorXd>(p, size);
  }
  return out;
}

Eigen::MatrixXd bayesmix::to_eigen(const Matrix &mat) {
  int nrow = mat.rows();
  int ncol = mat.cols();
  Eigen::MatrixXd out;
  if (nrow > 0 & ncol > 0) {
    const double *p = &(mat.data())[0];
    out = Eigen::Map<const Eigen::MatrixXd>(p, nrow, ncol);
  }
  return out;
}
