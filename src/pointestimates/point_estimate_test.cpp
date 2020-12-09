#include <iostream>
#include "../../lib/math/lib/eigen_3.3.7/Eigen/Dense"
#include "clustering/ClusteringProcess.hpp"

int main() {
  Eigen::MatrixXi mcmc(3, 5);  // T = 3
  std::cerr << "1" << std::endl;
  mcmc << 1, 1, 1, 2, 3, 1, 1, 2, 3, 3, 1, 1, 2, 2, 2;

  Eigen::VectorXi a(5);
  a << 1, 1, 2, 3, 3;

  std::cerr << "2" << std::endl;

  ClusteringProcess cp(mcmc, BINDER_LOSS);
  std::cerr << "3" << std::endl;

  double epl = cp.expected_posterior_loss(a);
  std::cerr << "4" << std::endl;

  std::cerr << "Expected posterior loss : " << epl << std::endl;
}