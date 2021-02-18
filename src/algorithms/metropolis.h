#ifndef SRC_ALGORITHMS_METROPOLIS_H
#define SRC_ALGORITHMS_METROPOLIS_H

#include <Eigen/Dense>

// Model: y_i | alpha ~ Bern( logit^-1(x'_i alpha) )
//              alpha ~ N(0, sig2 * I)
// MALA proposal density: h(x) = N(a_M, eta * I) with
//                         a_M = alpha + tau * grad(log(f(alpha|rest)))
// Here grad(...) = -Lambda alpha + sum_i (y_i logit^-1(x'_i alpha)) x_i

class Metropolis {
 protected:
  unsigned int iter;
  unsigned int maxiter = 1000;

  // DESIGN PARAMETERS
  //! Step size parameter aka tau
  double step = 0.05;
  //! Proposed variance aka eta
  double prop_var = 0.5;
  //!
  bool use_mala;

  // DATA
  Eigen::VectorXd data;
  Eigen::MatrixXd covariates;
  unsigned int dim;
  //! True variance aka sig2
  double true_var = 5.0;
  //! State aka alpha
  Eigen::VectorXd state;
  //! Acceptance probability ratio
  double logratio;

  // UTILITIES
  Eigen::VectorXd mala_mean() const;
  Eigen::VectorXd draw_proposal() const;
  double like_lpdf(const Eigen::VectorXd &alpha) const;
  double prior_lpdf(const Eigen::VectorXd &alpha) const;
  void metropolis_hastings_step();
  void output();

 public:
  Metropolis() = default;
  ~Metropolis() = default;

  double sigmoid(const double x) const { return 1.0 / (1.0 + std::exp(-x)); }

  void generate_data();

  void set_prop_var(const double var_) { prop_var = var_; }
  void set_true_var(const double var_) { true_var = var_; }
  void set_step(const double step_) { step = step_; }
  void set_data(const Eigen::VectorXd &data_) { data = data_; }
  void set_covariates(const Eigen::MatrixXd &covar_) { covariates = covar_; }

  void run(const bool use_mala_) {
    iter = 0;
    use_mala = use_mala_;
    dim = covariates.cols();
    state = Eigen::VectorXd::Zero(dim);
    while (iter < maxiter) {
      metropolis_hastings_step();
      iter++;
    }
  }
};

#endif  // SRC_ALGORITHMS_METROPOLIS_H
