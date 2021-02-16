#ifndef CREDIBLE_BALL_HPP
#define CREDIBLE_BALL_HPP

#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "../../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "BinderLoss.hpp"
#include "LossFunction.hpp"
#include "VariationInformation.hpp"

enum LOSS_FUNCTION {
  BINDER_LOSS,
  VARIATION_INFORMATION,
  VARIATION_INFORMATION_NORMALIZED
};

class CredibleBall {
 private:
  LossFunction* loss_function;     // metric to compute the region
  Eigen::MatrixXi mcmc_sample;     // MCMC matrix of clusters
  Eigen::VectorXi point_estimate;  // output of the greedy algorithm
  set<int, greater<int>>
      credibleBall;  // set of index clusters inside the credible region
  int T;             // number of clusters, ie mcmc_sample.rows
  int N;             // dimension of cluster, ie mcmc_sample.cols
  double alpha;      // level of the credible ball region
  double radius;     // radius of the credible ball

 public:
  CredibleBall(LOSS_FUNCTION loss_type_, Eigen::MatrixXi& mcmc_sample_,
               double alpha_, Eigen::VectorXi& point_estimate_);
  ~CredibleBall();
  double calculateRegion(double rate);   // calculate the radius
  Eigen::VectorXi VerticalUpperBound();  // index of cluusters of the VUB
  Eigen::VectorXi VerticalLowerBound();  // index of the clusters of the VLB
  Eigen::VectorXi HorizontalBound();     // index of the clusters of the HB

 private:
  void populateCredibleSet();  // populate the credibleBall
  int count_cluster_row(int index);
};
#endif  // CREDIBLE_BALL_HPP
