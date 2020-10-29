#ifndef CLUSTER_UTILS_HPP
#define CLUSTER_UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "../collectors/BaseCollector.hpp"
#include "../utils/proto_utils.hpp"

namespace bayesmix {

//! Estimates the clustering structure of the data via LS minimization
Eigen::VectorXi cluster_estimate(Eigen::MatrixXi allocation_chain);


//! Writes unique values of each datum in csv form
void write_clustering_to_file(
    const Eigen::VectorXi &best_clust,
    const std::string &filename = "resources/clust_best.csv");
}  // namespace bayesmix

#endif  // CLUSTER_UTILS_HPP
