#include "ClusteringProcess.hpp"
#include <iostream>

using namespace std;

ClusteringProcess::ClusteringProcess(Eigen::MatrixXi &mcmc_sample_,
                                     LOSS_FUNCTION loss_type)
    : loss_function(0)
{
  std::cout << "[CONSTRUCTORS]" << std::endl;
  std::cout << "ClusteringProcess Constructor" << std::endl;

  mcmc_sample = mcmc_sample_;
  T = mcmc_sample.rows();
  switch(loss_type) {
    case BINDER_LOSS: {
      loss_function = new BinderLoss(1,1);
      break;
    }

    default: ; // todo
  }
}

ClusteringProcess::~ClusteringProcess() {
  std::cout << std::endl << "[DESTRUCTORS]" << std::endl;
  std::cout << "Clustering Process destructor" << std::endl;
  delete loss_function;
}

double ClusteringProcess::expected_posterior_loss(Eigen::VectorXi a)
{
  double epl = 0;
  loss_function->SetFirstCluster(a);

  for (int t = 0; t < T; t++) {
    loss_function->SetSecondCluster(mcmc_sample.row(t));
    cout << loss_function << endl;
    epl += loss_function->Loss();
  }

  std::cout << "epl (not normalized)=" << epl << "; T=" << T << std::endl;
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