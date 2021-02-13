#include "ClusterEstimator.hpp"
using namespace std;

ClusterEstimator::ClusterEstimator(Eigen::MatrixXi &mcmc_sample_, LOSS_FUNCTION loss_type,
                  int Kup, Eigen::VectorXi &initial_partition_)
    : loss_function(0)
{
  mcmc_sample = mcmc_sample_;
  T = mcmc_sample.rows();
  N = mcmc_sample.cols();
  K_up = Kup;
  initial_partition= initial_partition_;
  switch(loss_type) {
    case BINDER_LOSS: loss_function = new BinderLoss(1,1); break;
    case VARIATION_INFORMATION: {
      loss_function = new VariationInformation(false);
      break;
    }
    case VARIATION_INFORMATION_NORMALIZED: {
      loss_function = new VariationInformation(true);
      break;
    }
    default:
      throw std::domain_error("Loss function not recognized");
  }
}

ClusterEstimator::~ClusterEstimator() {
  delete loss_function;
}

double ClusterEstimator::expected_posterior_loss(Eigen::VectorXi a)
{
  double epl = 0;
  loss_function->SetFirstCluster(a);

  for (int t = 0; t < T; t++) {
    loss_function->SetSecondCluster(mcmc_sample.row(t));
    epl += loss_function->Loss();
  }

  return epl / T;
}


Eigen::VectorXi ClusterEstimator::cluster_estimate(MINIMIZATION_METHOD method) {
  switch (method) {
    case GREEDY:
      return greedy_algorithm(initial_partition);

    default:
      throw std::domain_error("Non valid method chosen");
  }
}


/**
 * replace (vec[i], vec[i+1], ..., vec[N-1]) with (vec[i+1], ..., vec[N-1], ?)
 * and then resize
 */
void delete_ith_element(Eigen::VectorXi &vec, int i) {
  if (vec.size() == 0) {
    throw std::domain_error("Can't delete an element from empty vector");
  }
  if (i >= vec.size() or i < 0) {
    throw std::domain_error("Wrong index to delete element");
  }

  copy(vec.data() + i + 1, vec.data() + vec.size(), vec.data() + i);
  vec.conservativeResize(vec.size() - 1);
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

int argmin(Eigen::VectorXd &vec) {
  int argmin(0);
  double min(vec(0));
  for (int index = 0; index < vec.size(); index++) {
    if (vec(index) < min) {
      min = vec(index);
      argmin = index;
    }
  }
  return argmin;
}

/**
 * The output of the greedy algorithm is an estimate with random cluster labels
 * This function aims to reordonning and renaming these labels to have a
 * more readable result.
 *
 * @param cluster
 *
 * Example : cluster = [5, 5, 5, 17, 23, 5, 17]
 * should be modified to give : cluster = [1, 1, 1, 2, 3, 1, 2]
 */
void rename_labels(Eigen::VectorXi &cluster) {
  map<int, int> m;
  int current_index(1);
  for (int i = 0; i < cluster.size(); i ++) {
    if (m[cluster(i)] == 0) {
      m[cluster(i)] = current_index;
      cluster(i) = current_index;
      current_index++;
    } else {
      cluster(i) = m[cluster(i)];
    }
  }
}


/**
 * a starting partition
 */
Eigen::VectorXi ClusterEstimator::greedy_algorithm(Eigen::VectorXi &a) {
  double phi_a(expected_posterior_loss(a)), phi_stop;
  Eigen::VectorXi nu, a_modified;
  Eigen::VectorXd epl_vec(K_up  );
  bool stop = false;
  int cmpt = 0;

  while(!stop) {
    phi_stop = phi_a;
    nu = Eigen::VectorXi::LinSpaced(N, 1 ,N);
    while(nu.size() != 0) {
      int i = rand() % nu.size();  // random int between 0 and size-1
      int nu_i = nu(i);
      delete_ith_element(nu, i);
      for (int s = 1; s <= K_up; s++) {
        // ai_r->s means a[i] is now equal to s
        a_modified = a;
        a_modified(nu_i - 1) = s;
        epl_vec(s - 1) = expected_posterior_loss(a_modified);
      }
      a(nu_i - 1) = argmin(epl_vec) + 1;
      phi_a = expected_posterior_loss(a);
    }
    if (phi_stop == phi_a) {
      stop = true;
    }
    cmpt++;
  }

  rename_labels(a);
//  cout << endl << "FINAL CLUSTER : " << a.transpose() << endl;
  cout << mcmc_sample.rows() << ":" << mcmc_sample.cols() << " --> " << cmpt << " while loops." << endl;
  return a;
}

