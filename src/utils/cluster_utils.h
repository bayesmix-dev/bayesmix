#ifndef BAYESMIX_UTILS_CLUSTER_UTILS_H_
#define BAYESMIX_UTILS_CLUSTER_UTILS_H_

#include <stan/math/rev.hpp>

//! \file cluster_utils.h
//! The `cluster_utils.h` file includes some utilities for cluster estimation.
//! These functions only use Eigen objects.

namespace bayesmix {

//! Computes the posterior similarity matrix the data
Eigen::MatrixXd posterior_similarity(const Eigen::MatrixXd &alloc_chain);

//! Estimates the clustering structure of the data via LS minimization
Eigen::VectorXi cluster_estimate(const Eigen::MatrixXi &alloc_chain);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_CLUSTER_UTILS_H_
