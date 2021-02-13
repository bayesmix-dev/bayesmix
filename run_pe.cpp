#include <iostream>

#include "src/includes.hpp"

Eigen::VectorXd epl_for_each_Kup(Eigen::MatrixXi &mcmc, int loss_type) {
  int N = mcmc.cols();
  Eigen::VectorXd epl_vec(mcmc.cols());
  Eigen::VectorXi initial_partition =  Eigen::VectorXi::LinSpaced(N, 1 ,N);

  for (int K = 1; K < N; K++) {
    ClusterEstimator cp(mcmc,  static_cast<LOSS_FUNCTION>(loss_type),
                         K, initial_partition);
    epl_vec(K) = cp.expected_posterior_loss(
        cp.cluster_estimate(GREEDY));
  }

  return epl_vec;
}

int main(int argc, char *argv[]) {
  //std::cout << "Running run_pe.cpp" << std::endl;

  if (argc != 5) {
    throw domain_error("Syntax : ./run_pe filename_in filename_out loss Kup");
  }

  std::string filename_in = argv[1];
  std::string filename_out = argv[2];
  int loss_type = std::stoi(argv[3]);
  int Kup = std::stoi(argv[4]);
  Eigen::MatrixXi mcmc;
  mcmc = bayesmix::read_eigen_matrix(filename_in);
//  std::cout << "Matrix with dimensions : " << mcmc.rows()
  std::cout << "Matrix with dimensions : " << mcmc.rows()
            << "*" << mcmc.cols() << " found." << std::endl;

  // Compute epl for each Kup to see the best
  if (Kup == -1) {
    cout << "Computation of epl for each K" << endl;
    bayesmix::write_matrix_to_file(epl_for_each_Kup(mcmc, loss_type).transpose(),
                                      filename_out);
  }

  else if (Kup > 0) {
    Eigen::VectorXi initial_partition =  Eigen::VectorXi::LinSpaced(mcmc.cols(), 1 ,mcmc.cols());
    ClusterEstimator cp(mcmc, static_cast<LOSS_FUNCTION>(loss_type),
                         Kup,initial_partition);
    Eigen::VectorXi result = cp.cluster_estimate(GREEDY);
    bayesmix::write_matrix_to_file(result.transpose(),  filename_out);

//    Eigen::VectorXd true_result = bayesmix::cluster_estimate(mcmc);
//    cout << "True result : " << true_result.transpose() << endl;
  }

}

