#include "ClusteringProcess.hpp"

double ClusteringProcess::expected_posterior_loss(Eigen::VectorXi a)
{
  // todo
}

Eigen::VectorXi ClusteringProcess::cluster_estimate(MINIMIZATION_METHOD method) {
  switch (method) {
    case GREEDY:
      return greedy_algorithm();

    default: ;
      // todo
  }
}

Eigen::VectorXi ClusteringProcess::greedy_algorithm() {
  // todo
}