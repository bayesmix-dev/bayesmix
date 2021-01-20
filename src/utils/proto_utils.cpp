#include "proto_utils.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <Eigen/Dense>
#include <fstream>

#include "matrix.pb.h"

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

void bayesmix::read_proto_from_file(const std::string &filename,
                                    google::protobuf::Message *out) {
  std::ifstream ifs(filename);
  google::protobuf::io::IstreamInputStream iis(&ifs);
  auto success = google::protobuf::TextFormat::Parse(&iis, out);
  if (!success) {
    std::cout << "Error " << success << " in read_proto_from_file"
              << std::endl;
  }
}
