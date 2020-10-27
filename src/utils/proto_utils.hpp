#ifndef PROTO_UTILS
#define PROTO_UTILS

#include <Eigen/Dense>

#include "../collectors/chain_state.pb.h"

namespace bayesmix {
//! Turns a single unique value from Protobuf object form into a matrix
Eigen::MatrixXd proto_param_to_matrix(const Param &par);
}  // namespace bayesmix

#endif  // PROTO_UTILS
