#include "cluster_utils.h"

#include <Eigen/SparseCore>
#include <stan/math/rev.hpp>

#include "lib/progressbar/progressbar.h"
#include "proto_utils.h"

Eigen::MatrixXd bayesmix::posterior_similarity(
    const Eigen::MatrixXd &alloc_chain) {
  unsigned int n_data = alloc_chain.cols();
  Eigen::MatrixXd mean_diss = Eigen::MatrixXd::Zero(n_data, n_data);
  // Loop over pairs (i,j) of data points
  for (int i = 0; i < n_data; i++) {
    for (int j = 0; j < i; j++) {
      Eigen::ArrayXd diff = alloc_chain.col(i) - alloc_chain.col(j);
      mean_diss(i, j) = (diff == 0).count();
    }
  }
  return mean_diss / alloc_chain.rows();
}

Eigen::VectorXi bayesmix::cluster_estimate(
    const Eigen::MatrixXi &alloc_chain) {
  // Initialize objects
  unsigned n_iter = alloc_chain.rows();
  unsigned int n_data = alloc_chain.cols();
  std::vector<Eigen::SparseMatrix<double> > all_diss;
  progresscpp::ProgressBar bar(n_iter, 60);

  // Compute mean
  // std::cout << "(Computing mean dissimilarity... " << std::flush;
  Eigen::MatrixXd mean_diss =
      bayesmix::posterior_similarity(alloc_chain.cast<double>());
  // std::cout << "Done)" << std::endl;

  // Compute Frobenius norm error of all iterations
  // std::cout << "Computing Frobenius norm error... " << std::endl;
  Eigen::VectorXd errors(n_iter);
  for (int k = 0; k < n_iter; k++) {
    for (int i = 0; i < n_data; i++) {
      for (int j = 0; j < i; j++) {
        double x = (alloc_chain(k, i) == alloc_chain(k, j));
        errors(k) += (x - mean_diss(i, j)) * (x - mean_diss(i, j));
      }
    }
    // Progress bar
    // ++bar;
    // bar.display();
  }
  // bar.done();
  // std::cout << "Done" << std::endl;  // Print Ending Message

  // Find iteration with the least error
  std::ptrdiff_t ibest;
  unsigned int min_err = errors.minCoeff(&ibest);
  return alloc_chain.row(ibest).transpose();
}
