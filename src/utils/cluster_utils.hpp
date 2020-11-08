#ifndef BAYESMIX_UTILS_CLUSTER_UTILS_HPP_
#define BAYESMIX_UTILS_CLUSTER_UTILS_HPP_

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "../utils/proto_utils.hpp"

namespace bayesmix {
//! Estimates the clustering structure of the data via LS minimization
Eigen::VectorXi cluster_estimate(Eigen::MatrixXi allocation_chain);
}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_CLUSTER_UTILS_HPP_
