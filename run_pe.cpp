#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run_pe.cpp" << std::endl;

  if (argc != 4) {
    throw domain_error("Syntax : ./run_ce filename_in filename_out loss");
  }

  std::string filename_in = argv[1];
  std::string filename_out = argv[2];
  int loss_type = std::stoi(argv[3]);
  Eigen::MatrixXi mcmc;
  mcmc = bayesmix::read_eigen_matrix(filename_in);
  std::cout << "Matrix with dimensions : " << mcmc.rows()
            << "*" << mcmc.cols() << " found." << std::endl;

  ClusteringProcess cp(mcmc, static_cast<LOSS_FUNCTION>(loss_type));
  Eigen::VectorXi result = cp.cluster_estimate(GREEDY);

  bayesmix::write_matrix_to_file(result.transpose(),  filename_out);
}