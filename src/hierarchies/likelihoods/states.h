#ifndef BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_

#include <Eigen/Dense>

namespace State {

struct UniLS {
  double mean, var;
};

struct MultiLS {
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec, prec_chol;
  double prec_logdet;
};

struct UniLinReg {
  Eigen::VectorXd regression_coeffs;
  double var;
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
