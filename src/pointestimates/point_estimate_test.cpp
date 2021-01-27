#include <iostream>
#include "../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "clustering/ClusteringProcess.hpp"

int main() {
  Eigen::MatrixXi mcmc(3, 5);  // T = 3
  mcmc << 1, 1, 1, 2, 3,
          1, 1, 2, 3, 3,
          1, 1, 2, 2, 2;

  ClusteringProcess cp_binder(mcmc, BINDER_LOSS);

  cp_binder.cluster_estimate(GREEDY);

  ClusteringProcess cp_vi(mcmc, VARIATION_INFORMATION);
  cp_vi.cluster_estimate(GREEDY);

  ClusteringProcess cp_vi_normalized(mcmc, VARIATION_INFORMATION_NORMALIZED);
  cp_vi_normalized.cluster_estimate(GREEDY);
}

//int main() {
//  Eigen::MatrixXi mcmc;  // T = 3
//  mcmc << read_eigen_matrix("test.csv");
//
//  cout << mcmc;
//
//
//  Eigen::VectorXi a(5);
//  a << 1, 1, 2, 3, 3;
//
//  ClusteringProcess cp(mcmc, BINDER_LOSS);
//  //double epl = cp.expected_posterior_loss(a);
//  //std::cout << "Expected posterior loss : " << epl << std::endl;
//
//  cp.cluster_estimate(GREEDY);
//}
//

