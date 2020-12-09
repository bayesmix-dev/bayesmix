#include "ClusteringProcess.hpp"
#include <iostream>

ClusteringProcess::ClusteringProcess(Eigen::MatrixXi &mcmc_sample_, LOSS_FUNCTION loss_type) {
  mcmc_sample = mcmc_sample_;
  T = mcmc_sample.rows();
  std::cerr << "2.1" << std::endl;
  switch(loss_type) {
    case BINDER_LOSS: {
      BinderLoss tmp;
      std::cerr << "2.2" << std::endl;
      loss_function = &tmp;
      std::cerr << "2.3" << std::endl;
      break;
    }

    default: ; // todo
  }
  std::cerr << "2.4" << std::endl;
}

double ClusteringProcess::expected_posterior_loss(Eigen::VectorXi a)
{
  double epl = 0;
  std::cout << "3.1" << std::endl;
  loss_function->SetFirstCluster(a);
  std::cout << "3.2" << std::endl;

  for (int t = 0; t < T; t++) {
    loss_function->SetSecondCluster(a);
    loss_function->SetSecondCluster(mcmc_sample.row(t));
    epl += loss_function->Loss();
  }
  std::cout << "3.3" << std::endl;
  std::cout << "epl=" << epl << "; T=" << T << std::endl;

  return epl / T;
}

Eigen::VectorXi ClusteringProcess::cluster_estimate(MINIMIZATION_METHOD method) {
  switch (method) {
    case GREEDY:
      return greedy_algorithm();

    default:
      throw std::domain_error("Non valid method chosen");
  }
}

Eigen::VectorXi ClusteringProcess::greedy_algorithm() {
  // todo
  return Eigen::VectorXi(3);
}