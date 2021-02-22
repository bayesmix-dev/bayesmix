#ifndef CREDIBLE_BALL_HPP
#define CREDIBLE_BALL_HPP

#include <cstdlib>
#include <iostream>
#include <set>

#include <Eigen/Dense>
#include "../lossfunction/LossFunction.hpp"
#include "../lossfunction/BinderLoss.hpp"
#include "../lossfunction/VariationInformation.hpp"


class CredibleBall {
 private:
  LossFunction* loss_function;     // metric to compute the region
  Eigen::MatrixXi mcmc_sample;     // MCMC matrix of clusters
  Eigen::VectorXi point_estimate;  // output of the greedy algorithm
  set<int, greater<int>>
      credibleBall;     // set of index clusters inside the credible region
  int T;                // number of clusters, ie mcmc_sample.rows
  int N;                // dimension of cluster, ie mcmc_sample.cols
  double alpha;         // level of the credible ball region
  double radius;        // radius of the credible ball
  double prob;          // estimated probability
  double vlb_distance;  // distance to the vertical lower bounds
  double vub_distance;  // distance to the vertical upper bounds
  double hb_distance;   // distance to the horizontal bounds

 public:
  CredibleBall(LOSS_FUNCTION loss_type_, Eigen::MatrixXi& mcmc_sample_,
               double alpha_, Eigen::VectorXi& point_estimate_);
  ~CredibleBall();

  void calculateRegion(
      double rate);    // calculate the points in the credible region
  double getRadius();  // returns the calculated radius
  Eigen::VectorXi VerticalUpperBound();  // index of cluusters of the VUB
  Eigen::VectorXi VerticalLowerBound();  // index of the clusters of the VLB
  Eigen::VectorXi HorizontalBound();     // index of the clusters of the HB
  void sumary(
      Eigen::VectorXi HB, Eigen::VectorXi VUB, Eigen::VectorXi VLB,
      string filename);  // generates a file with the bounds information

 protected:
  void populateCredibleSet();  // populate the credibleBall
  int count_cluster_row(int index);
  int getMinimalCrdinality();
  int getMaximalCardinality();
  double getMaxDistance(set<int, greater<int>> vec1);
  double getMinDistance();
  void saveSelectedClusters(set<int, greater<int>>& vec1, double& max_distance,
                            set<int, greater<int>>& vlb);
};
#endif  // CREDIBLE_BALL_HPP
