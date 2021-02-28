#ifndef BAYESMIX_CLUSTERESTIMATOR_HPP
#define BAYESMIX_CLUSTERESTIMATOR_HPP

#include "lossfunction/LossFunction.hpp"
#include "lossfunction/BinderLoss.hpp"
#include "lossfunction/VariationInformation.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <map>


// in case we want to add other minimization methods in the future.
enum MINIMIZATION_METHOD {
  GREEDY
};


/**
 * Describe the full process of clustering :
 *      - choice of loss function via LossFunction member
 *      - Choice of minimization method via loss_type in the constructor
 *      - computation of cluster estimate through cluster_estimate method.
 */

class ClusterEstimator {
 private:
  LossFunction* loss_function;
  Eigen::MatrixXi mcmc_sample; // T*N matrix
  int T; // total time of the process
  int N;
  int K_up;
  Eigen::VectorXi initial_partition;
 public:
  ClusterEstimator(Eigen::MatrixXi &mcmc_sample_, LOSS_FUNCTION loss_type_,
                    int K_up, Eigen::VectorXi &initial_partition_);
  ~ClusterEstimator();
  double expected_posterior_loss(Eigen::VectorXi a);
//  Eigen::VectorXd expected_posterior_loss_for_each_Kup(Eigen::VectorXi a);
  Eigen::VectorXi cluster_estimate(MINIMIZATION_METHOD method);
  Eigen::VectorXi greedy_algorithm(Eigen::VectorXi &a);
};

#endif  // BAYESMIX_CLUSTERESTIMATOR_HPP
