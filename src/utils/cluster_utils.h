#ifndef BAYESMIX_UTILS_CLUSTER_UTILS_H_
#define BAYESMIX_UTILS_CLUSTER_UTILS_H_

#include <Eigen/Dense>

//! This file includes some utilities for cluster estimation. These functions
//! only use Eigen ojects.

namespace bayesmix {
//! Computes the posterior similarity matrix the data
Eigen::MatrixXd posterior_similarity(const Eigen::MatrixXd &alloc_chain);

//! Estimates the clustering structure of the data via LS minimization
Eigen::VectorXd cluster_estimate(const Eigen::MatrixXd &alloc_chain);
}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_CLUSTER_UTILS_H_
