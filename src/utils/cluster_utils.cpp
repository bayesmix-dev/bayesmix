#include "cluster_utils.hpp"

//! \param coll Collector containing the algorithm chain
//! \return     Index of the iteration containing the best estimate
State bayesmix::cluster_estimate(BaseCollector *coll) {
  // Read chain from collector
  std::deque<State> chain = coll->get_chain();

  // Initialize objects
  unsigned n_iter = chain.size();
  unsigned int n = chain[0].allocations_size();
  Eigen::VectorXd errors(n_iter);
  Eigen::MatrixXd mean_diss = Eigen::MatrixXd::Zero(n, n);
  std::vector<Eigen::SparseMatrix<double> > all_diss;
  State temp;

  // Loop over iterations
  for (size_t i = 0; i < n_iter; i++) {
    // Find and all nonzero entries of the dissimilarity matrix
    std::vector<Eigen::Triplet<double> > triplets_list;
    triplets_list.reserve(n * n / 4);
    for (size_t j = 0; j < n; i++) {
      for (size_t k = 0; k < j; k++) {
        if (chain[i].allocations(j) == chain[i].allocations(k)) {
          triplets_list.push_back(Eigen::Triplet<double>(j, k, 1.0));
        }
      }
    }
    // Build dissimilarity matrix and update total dissimilarity
    Eigen::SparseMatrix<double> dissim(n, n);
    dissim.setZero();
    dissim.setFromTriplets(triplets_list.begin(), triplets_list.end());
    all_diss.push_back(dissim);
    mean_diss += dissim;
  }
  // Average over iterations
  mean_diss = mean_diss / n_iter;

  // Compute Frobenius norm error of all iterations
  for (size_t i = 0; i < n_iter; i++) {
    errors(i) = (mean_diss - all_diss[i]).norm();
  }

  // Find iteration with the least error
  std::ptrdiff_t ibest;
  unsigned int min_err = errors.minCoeff(&ibest);
  return chain[ibest];
}

//! \param filename Name of file to write to
void bayesmix::write_clustering_to_file(
    const State &best_clust,
    const std::string &filename /*= "resources/clust_best.csv"*/) {
  // Open file
  std::ofstream file;
  file.open(filename);

  // Loop over allocations vector of the saved best clustering
  for (size_t i = 0; i < best_clust.allocations_size(); i++) {
    unsigned int ci = best_clust.allocations(i);
    // Write allocation to file
    file << ci << ",";
    // Loop over unique values vector
    for (size_t j = 0; j < best_clust.uniquevalues(ci).params_size(); j++) {
      Eigen::MatrixXd temp_param(bayesmix::proto_param_to_matrix(
          best_clust.uniquevalues(ci).params(j)));
      for (size_t k = 0; k < temp_param.rows(); k++) {
        for (size_t h = 0; h < temp_param.cols(); h++) {
          // Write unique value to file
          if (h == temp_param.cols() - 1 && k == temp_param.rows() - 1 &&
              j == best_clust.uniquevalues(ci).params_size() - 1) {
            file << temp_param(k, h);
          } else {
            file << temp_param(k, h) << ",";
          }
        }
      }
    }
    file << std::endl;
  }
  file.close();
  std::cout << "Successfully wrote clustering to " << filename << std::endl;
}
