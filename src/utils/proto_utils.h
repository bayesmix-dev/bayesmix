#ifndef BAYESMIX_UTILS_PROTO_UTILS_H_
#define BAYESMIX_UTILS_PROTO_UTILS_H_

#include <Eigen/Dense>

#include "matrix.pb.h"

//! This file implements a few useful functions to manipulate Protobuf objects.
//! For instance, this library implements its own version of vectors and
//! matrices, and the functions implemented here convert from these types to
//! the Eigen ones and viceversa. One can also read a Protobuf from a text
//! file. This is mostly useful for algorithm configuration files.

namespace bayesmix {

//! Writes an Eigen vector to a bayesmix::Vector Protobuf object by pointer
void to_proto(const Eigen::VectorXd &vec, bayesmix::Vector *out);

//! Writes an Eigen matrix to a bayesmix::Matrix Protobuf object by pointer
void to_proto(const Eigen::MatrixXd &mat, bayesmix::Matrix *out);

//! Converts a bayesmix::Vector Protobuf object into an Eigen vector
Eigen::VectorXd to_eigen(const bayesmix::Vector &vec);

//! Converts a bayesmix::Matrix Protobuf object into an Eigen matrix
Eigen::MatrixXd to_eigen(const bayesmix::Matrix &mat);

//! Writes from a given file to a Protobuf object via pointer
void read_proto_from_file(const std::string &filename,
                          google::protobuf::Message *out);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_PROTO_UTILS_H_
