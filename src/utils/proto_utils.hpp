#ifndef BAYESMIX_UTILS_PROTO_UTILS_HPP_
#define BAYESMIX_UTILS_PROTO_UTILS_HPP_

#include <Eigen/Dense>

#include "../../proto/cpp/matrix.pb.h"

namespace bayesmix {
void to_proto(const Eigen::MatrixXd &mat, Matrix *out);
void to_proto(const Eigen::VectorXd &vec, Vector *out);

Eigen::VectorXd to_eigen(const Vector &vec);
Eigen::MatrixXd to_eigen(const Matrix &mat);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_PROTO_UTILS_HPP_
