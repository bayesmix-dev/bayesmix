#ifndef CLUSTER_UTILS_HPP
#define CLUSTER_UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "../utils/proto_utils.hpp"

namespace bayesmix {

//! Estimates the clustering structure of the data via LS minimization
Eigen::VectorXi cluster_estimate(Eigen::MatrixXi allocation_chain);
}  // namespace bayesmix

#endif  // CLUSTER_UTILS_HPP
