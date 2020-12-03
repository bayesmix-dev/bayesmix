#ifndef BAYESMIX_CLUSTERINGPROCESS_HPP
#define BAYESMIX_CLUSTERINGPROCESS_HPP

#include "../lossfunction/LossFunction.hpp"
#include <../../../lib/math/lib/eigen_3.3.7/Eigen/Dense>

// in case we want to add other minimization methods in the future.
enum MINIMIZATION_METHOD {
  GREEDY
};

class ClusteringProcess
{
  private:
      LossFunction *loss_function;
      Eigen::MatrixXi * mcmc_sample;

   public:
      double expected_posterior_loss(Eigen::VectorXi a);
      Eigen::VectorXi cluster_estimate(MINIMIZATION_METHOD method);
      Eigen::VectorXi greedy_algorithm();
};

#endif  // BAYESMIX_CLUSTERINGPROCESS_HPP
