#ifndef BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_

#include <Eigen/Dense>

namespace State {

class Base {
 protected:
  Base() = default;

 public:
  virtual ~Base() = default;
};

struct UniLS : public Base {
  double mean, var;
};

struct MultiLS : public Base {
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec, prec_chol;
  double prec_logdet;
};

struct UniLinReg : public Base {
  Eigen::VectorXd regression_coeffs;
  double var;
};

}  // namespace State

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOOD_STATES_H_
