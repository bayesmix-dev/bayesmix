#include "ClusteringProcess.hpp"
#include <iostream>
#include <stdlib.h>

using namespace std;

ClusteringProcess::ClusteringProcess(Eigen::MatrixXi &mcmc_sample_,
                                     LOSS_FUNCTION loss_type)
    : loss_function(0)
{
  std::cout << "[CONSTRUCTORS]" << std::endl;
  std::cout << "ClusteringProcess Constructor" << std::endl;

  mcmc_sample = mcmc_sample_;
  T = mcmc_sample.rows();
  N = mcmc_sample.cols();
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
    epl += loss_function->Loss();
  }

  std::cout << "epl (not normalized)=" << epl << "; T=" << T << std::endl;
  return epl / T;
}

Eigen::VectorXi ClusteringProcess::cluster_estimate(MINIMIZATION_METHOD method) {
  Eigen::VectorXi a(5);
  a << 1, 1, 2, 3, 3;
  switch (method) {
    case GREEDY:
      return greedy_algorithm(a);

    default:
      throw std::domain_error("Non valid method chosen");
  }
}


/**
 * replace (vec[i], vec[i+1], ..., vec[N-1]) with (vec[i+1], ..., vec[N-1], ?)
 * and then resize
 */
void delete_ith_element(Eigen::VectorXi &vec, int i) {
  copy(vec.data() + i + 1, vec.data() + vec.size(), vec.data() + i);
  cout << "vec copy = " << vec.transpose() << "; ";
  vec.conservativeResize(vec.size() - 1);
  cout << "vec resized = " << vec.transpose() << endl;

}

/**
 * Return the index of the max coefficient in vec
 * No such method seems to exist in Eigen lib
 */
int argmax(Eigen::VectorXd &vec) {
  int argmax(0);
  double max(0);
  for (int index = 0; index < vec.size(); index++) {
    if (vec(index) > max) {
      max = vec(index);
      argmax = index;
    }
  }
  return argmax;
}

/**
 * a starting partition
 */
Eigen::VectorXi ClusteringProcess::greedy_algorithm(Eigen::VectorXi &a) {
  double phi_a(expected_posterior_loss(a)), phi_stop;
  Eigen::VectorXi nu, a_modified;
  Eigen::VectorXd epl_vec(N);
  bool stop = false;
  while(!stop) {
    phi_stop = phi_a;
    nu = Eigen::VectorXi::LinSpaced(N, 1 ,N);
    cout << "initial nu=" << nu.transpose() << endl;
    while(nu.size() != 0) {
      cout << endl;
      int i = rand() % nu.size(); // random int between 0 and size-1
      cout << "[i=" << i << "; current Nu=" << nu.transpose() << "]" << endl;
      delete_ith_element(nu, i);
      for (int s = 1; s < N; s++) {  // Kup = N, could be changed
        // ai_r->s means a[i] is now equal to s
        cout << "    [s=" << s << "]" << endl;
        a_modified = a;
        a_modified(i) = s;
        cout << "    a_modified=" << a_modified.transpose() << endl;
        cout << "    ";
        epl_vec(s) = expected_posterior_loss(a_modified);
      }
      a(i) = argmax(epl_vec);
      cout << "--> Final ";
      phi_a = expected_posterior_loss(a);

      if (phi_stop == phi_a) {
        stop = true;
      }
    }
  }

  cout << endl << "FINAL CLUSTER : " << a.transpose() << endl;
  return a;
}

