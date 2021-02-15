#ifndef SRC_ALGORITHMS_METROPOLIS_H
#define SRC_ALGORITHMS_METROPOLIS_H

#include <Eigen/Dense>

#include "src/utils/rng.h"

class Metropolis {
 protected:
  unsigned int iter = 0;
  unsigned int maxiter = 5000;
  auto &rng = bayesmix::Rng::Instance().get();
  //! Design parameters
  double var = 1.0;
  double tau = 1.0;
  bool use_mala;
  //! State
  unsigned int dim;
  Eigen::VectorXd state;

  Eigen::VectorXd standard_mean() const;
  Eigen::VectorXd mala_mean() const;
  void metropolis_hastings_step();
  void output();

 public:
  Metropolis() = default;
  ~Metropolis() = default;

  void set_dim(const unsigned int dim_) {
    dim = dim_;
    state = Eigen::VectorXd::Zero(dim);
  }

  void set_var(const double var_) { var = var_; }

  void run(const bool use_mala_) {
    use_mala = use_mala_;
    while(iter < maxiter) {
      metropolis_hastings_step();
      output();
      iter++;
    }
  }
};

#endif // SRC_ALGORITHMS_METROPOLIS_H
