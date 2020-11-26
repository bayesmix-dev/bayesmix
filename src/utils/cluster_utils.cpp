#include "cluster_utils.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "proto_utils.hpp"

//! \param coll Collector containing the algorithm chain
//! \return     Index of the iteration containing the best estimate
Eigen::VectorXd bayesmix::cluster_estimate(
    const Eigen::MatrixXd &alloc_chain) {
  // Initialize objects
  unsigned n_iter = alloc_chain.rows();
  unsigned int n_data = alloc_chain.cols();
  Eigen::MatrixXd mean_diss = Eigen::MatrixXd::Zero(n_data, n_data);
  std::vector<Eigen::SparseMatrix<double> > all_diss;

  // Loop over iterations
  for (size_t i = 0; i < n_iter; i++) {
    // Find and all nonzero entries of the dissimilarity matrix
    std::vector<Eigen::Triplet<double> > triplets_list;
    triplets_list.reserve(n_data * n_data / 4);
    for (size_t j = 0; j < n_data; i++) {
      for (size_t k = 0; k < j; k++) {
        if (alloc_chain(i, j) == alloc_chain(i, k)) {
          triplets_list.push_back(Eigen::Triplet<double>(j, k, 1.0));
        }
      }
    }
    // Build dissimilarity matrix and update total dissimilarity
    Eigen::SparseMatrix<double> dissim(n_data, n_data);
    dissim.setZero();
    dissim.setFromTriplets(triplets_list.begin(), triplets_list.end());
    all_diss.push_back(dissim);
    mean_diss += dissim;
  }
  // Average over iterations
  mean_diss = mean_diss / n_iter;

  // Compute Frobenius norm error of all iterations
  Eigen::VectorXd errors(n_iter);
  for (size_t i = 0; i < n_iter; i++) {
    errors(i) = (mean_diss - all_diss[i]).norm();
  }

  // Find iteration with the least error
  std::ptrdiff_t ibest;
  unsigned int min_err = errors.minCoeff(&ibest);
  return alloc_chain.row(ibest).transpose();
}
