#include <iostream>
#include "../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "clustering/ClusteringProcess.hpp"

int main() {
  Eigen::MatrixXi mcmc(3, 5);  // T = 3
  mcmc << 1, 1, 1, 2, 3,
      1, 1, 2, 3, 3,
      1, 1, 2, 2, 2;

  Eigen::VectorXi a(5);
  a << 1, 1, 2, 3, 3;

  ClusteringProcess cp(mcmc, BINDER_LOSS);
  //double epl = cp.expected_posterior_loss(a);
  //std::cout << "Expected posterior loss : " << epl << std::endl;

  cp.cluster_estimate(GREEDY);
}