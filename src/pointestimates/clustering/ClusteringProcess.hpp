#ifndef BAYESMIX_CLUSTERINGPROCESS_HPP
#define BAYESMIX_CLUSTERINGPROCESS_HPP

#include "../lossfunction/LossFunction.hpp"
#include "../lossfunction/BinderLoss.hpp"
#include "../../../lib/math/lib/eigen_3.3.7/Eigen/Dense"

// in case we want to add other minimization methods in the future.
enum MINIMIZATION_METHOD {
  GREEDY
};

enum LOSS_FUNCTION {
  BINDER_LOSS,
  VARIATION_INFORMATION,
  VARIATION_INFORMATION_NORMALIZED,
};

class ClusteringProcess
{
  private:
      LossFunction* loss_function;
      Eigen::MatrixXi mcmc_sample;
      int T; // total time of the process
   public:
      ClusteringProcess(Eigen::MatrixXi &mcmc_sample_, LOSS_FUNCTION loss_type);
      ~ClusteringProcess();
      double expected_posterior_loss(Eigen::VectorXi a);
      Eigen::VectorXi cluster_estimate(MINIMIZATION_METHOD method);
      Eigen::VectorXi greedy_algorithm();
};

#endif  // BAYESMIX_CLUSTERINGPROCESS_HPP
