#ifndef SRC_ALGORITHMS_METROPOLIS_H
#define SRC_ALGORITHMS_METROPOLIS_H

#include <Eigen/Dense>

// Model: y_i | alpha ~ Bern( logit^-1(x'_i alpha) )
//              alpha ~ N(0, Lambda^-1)
//          Lambda^-1 = sig2 * I
// MALA proposal density: h(x) = N(a_M, eta * I) with
//                         a_M = alpha + tau * grad(log(f(alpha|rest)))
// Here grad(...) = -Lambda alpha + sum_i (y_i logit^-1(x'_i alpha)) x_i

class Metropolis {
 protected:
  unsigned int iter;
  unsigned int maxiter = 1000;

  // DESIGN PARAMETERS
  //! Penalization parameter aka tau
  double penal = 0.1;
  //! Proposed variance aka eta
  double prop_var = 2.0;
  //!
  bool use_mala;

  // DATA
  Eigen::VectorXd data;
  Eigen::MatrixXd covariates;
  unsigned int dim;
  //! True variance aka sig2
  double true_var = 1.0;
  //! State aka alpha
  Eigen::VectorXd state;

  // UTILITIES
  Eigen::VectorXd standard_mean() const;
  Eigen::VectorXd mala_mean() const;
  void metropolis_hastings_step();
  void output();

 public:
  Metropolis() = default;
  ~Metropolis() = default;

  double inv_logit(const double x) const {
    return std::exp(x) / (1 + std::exp(x));
  }

  void set_dim(const unsigned int dim_) {
    dim = dim_;
    state = Eigen::VectorXd::Zero(dim);
  }
  void set_prop_var(const double var_) { prop_var = var_; }
  void set_true_var(const double var_) { true_var = var_; }
  void set_penal(const double penal_) { penal = penal_; }
  void set_data(const Eigen::VectorXd &data_) { data = data_; }
  void set_covariates(const Eigen::MatrixXd &covar_) { covariates = covar_; }

  void run(const bool use_mala_) {
    iter = 0;
    use_mala = use_mala_;
    dim = covariates.cols();
    state = Eigen::VectorXd::Zero(dim);  // TODO?
    while (iter < maxiter) {
      metropolis_hastings_step();
      // output();
      iter++;
    }
  }
};

#endif  // SRC_ALGORITHMS_METROPOLIS_H
