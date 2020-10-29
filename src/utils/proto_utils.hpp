#ifndef PROTO_UTILS
#define PROTO_UTILS

#include <Eigen/Dense>

#include "../../proto/cpp/matrix.pb.h"

namespace bayesmix {
void to_proto(const Eigen::MatrixXd &mat, Matrix *out);

void to_proto(const Eigen::VectorXd &vec, Vector *out);

Eigen::VectorXd to_eigen(const Vector &vec);

Eigen::MatrixXd to_eigen(const Matrix &vec);

}  // namespace bayesmix

#endif  // PROTO_UTILS
