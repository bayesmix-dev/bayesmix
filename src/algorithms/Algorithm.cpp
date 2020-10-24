#include "Algorithm.hpp"

//! \param iter Number of the current iteration
//! \return     Protobuf-object version of the current state
State Algorithm::get_state_as_proto(unsigned int iter) {
  // Transcribe allocations vector
  State iter_out;
  *iter_out.mutable_allocations() = {allocations.begin(), allocations.end()};

  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    UniqueValues uniquevalues_temp;
    for (size_t k = 0; k < unique_values[i]->get_state().size(); k++) {
      Eigen::MatrixXd par_temp = unique_values[i]->get_state()[k];
      Param par_temp_proto;
      for (size_t j = 0; j < par_temp.cols(); j++) {
        Par_Col col_temp;
        for (size_t h = 0; h < par_temp.rows(); h++) {
          col_temp.add_elems(par_temp(h, j));
        }
        par_temp_proto.add_par_cols();
        *par_temp_proto.mutable_par_cols(j) = col_temp;
      }
      uniquevalues_temp.add_params();
      *uniquevalues_temp.mutable_params(k) = par_temp_proto;
    }
    iter_out.add_uniquevalues();
    *iter_out.mutable_uniquevalues(i) = uniquevalues_temp;
  }
  return iter_out;
}

//! \param un_val Unique value in Protobuf-object form
//! \return       Matrix version of un_val
Eigen::MatrixXd Algorithm::proto_param_to_matrix(const Param &un_val) const {
  Eigen::MatrixXd par_matrix = Eigen::MatrixXd::Zero(
      un_val.par_cols(0).elems_size(), un_val.par_cols_size());

  // Loop over unique values to copy them one at a time
  for (size_t h = 0; h < un_val.par_cols_size(); h++) {
    for (size_t j = 0; j < un_val.par_cols(h).elems_size(); j++) {
      par_matrix(j, h) = un_val.par_cols(h).elems(j);
    }
  }
  return par_matrix;
}

void Algorithm::print_ending_message() const {
  std::cout << "Done" << std::endl;
}

//! \param coll Collector containing the algorithm chain
//! \return     Index of the iteration containing the best estimate
unsigned int Algorithm::cluster_estimate(BaseCollector *coll) {
  // Read chain from collector
  std::deque<State> chain = coll->get_chain();

  // Initialize objects
  unsigned n_iter = chain.size();
  unsigned int n = chain[0].allocations_size();
  Eigen::VectorXd errors(n_iter);
  Eigen::MatrixXd tot_diss = Eigen::MatrixXd::Zero(n, n);
  std::vector<Eigen::SparseMatrix<double> > all_diss;
  State temp;

  // Loop over iterations
  for (size_t h = 0; h < n_iter; h++) {
    // Find and all nonzero entries of the dissimilarity matrix
    std::vector<Eigen::Triplet<double> > triplets_list;
    triplets_list.reserve(n * n / 4);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < i; j++) {
        if (chain[h].allocations(i) == chain[h].allocations(j)) {
          triplets_list.push_back(Eigen::Triplet<double>(i, j, 1.0));
        }
      }
    }
    // Build dissimilarity matrix and update total dissimilarity
    Eigen::SparseMatrix<double> dissim(n, n);
    dissim.setZero();
    dissim.setFromTriplets(triplets_list.begin(), triplets_list.end());
    all_diss.push_back(dissim);
    tot_diss += dissim;
  }
  // Average over iterations
  tot_diss = tot_diss / n_iter;

  // Compute Frobenius norm error of all iterations
  for (size_t h = 0; h < n_iter; h++) {
    errors(h) = (tot_diss - all_diss[h]).norm();
  }

  // Find iteration with the least error
  std::ptrdiff_t i;
  unsigned int min_err = errors.minCoeff(&i);
  best_clust = chain[i];
  std::cout << "Optimal clustering: at iteration " << i << " with "
            << best_clust.uniquevalues_size() << " clusters" << std::endl;
  // Update flag
  clustering_was_computed = true;

  return i;
}

//! \param filename Name of file to write to
void Algorithm::write_clustering_to_file(const std::string &filename) const {
  if (!clustering_was_computed) {
    std::cerr << "Error: cannot write clustering to file; "
              << "cluster_estimate() must be called first" << std::endl;
    return;
  }

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
      Eigen::MatrixXd temp_param(
          proto_param_to_matrix(best_clust.uniquevalues(ci).params(j)));
      for (size_t k = 0; k < temp_param.rows(); k++) {
        for (size_t z = 0; z < temp_param.cols(); z++) {
          // Write unique value to file
          if (z == temp_param.cols() - 1 && k == temp_param.rows() - 1 &&
              j == best_clust.uniquevalues(ci).params_size() - 1) {
            file << temp_param(k, z);
          } else {
            file << temp_param(k, z) << ",";
          }
        }
      }
    }
    file << std::endl;
  }
  file.close();
  std::cout << "Successfully wrote clustering to " << filename << std::endl;
}

//! \param filename Name of file to write to
void Algorithm::write_density_to_file(const std::string &filename) const {
  if (!density_was_computed) {
    std::cerr << "Error: cannot write density to file; eval_density() "
              << "must be called first" << std::endl;
    return;
  }

  // Open file
  std::ofstream file;
  file.open(filename);

  // Loop over grid points
  for (size_t i = 0; i < density.first.rows(); i++) {
    Eigen::VectorXd point = density.first.row(i);
    // Write point coordinates
    for (size_t j = 0; j < point.size(); j++) {
      file << point(j) << ",";
    }
    // Write density value
    file << density.second(i) << std::endl;
  }

  file.close();
  std::cout << "Successfully wrote density to " << filename << std::endl;
}

void Algorithm::set_data(const std::string &filename){
  data = read_eigen_matrix(filename);
}
