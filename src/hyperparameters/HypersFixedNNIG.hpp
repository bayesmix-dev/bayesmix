#ifndef HYPERSFIXEDNNIG_HPP
#define HYPERSFIXEDNNIG_HPP

#include <cassert>

//! Class that represents fixed hyperparameters for an NNIG hierarchy.

//! That is, it represents hyperparameters without a prior distribution. It can
//! be used as a template argument for the univariate template class,
//! HierarchyNNIG. All constructors and setters have validity checks for the
//! inserted values.

class HypersFixedNNIG {
 protected:
  // HYPERPARAMETERS
  double mu0, lambda, alpha0, beta0;

  //! Raises error if the hypers values are not valid w.r.t. their own domain
  void check_hypers_validity() {
    assert(lambda > 0);
    assert(alpha0 > 0);
    assert(beta0 > 0);
  }

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~HypersFixedNNIG() = default;
  HypersFixedNNIG() = default;
  HypersFixedNNIG(const double mu0_, const double lambda_, const double alpha0_,
                  const double beta0_)
      : mu0(mu0_), lambda(lambda_), alpha0(alpha0_), beta0(beta0_) {
    check_hypers_validity();
  }

  // GETTERS AND SETTERS
  double get_mu0() const { return mu0; }
  double get_alpha0() const { return alpha0; }
  double get_beta0() const { return beta0; }
  double get_lambda() const { return lambda; }
  void set_mu0(const double mu0_) { mu0 = mu0_; }
  void set_alpha0(const double alpha0_) {
    assert(alpha0_ > 0);
    alpha0 = alpha0_;
  }
  void set_beta0(const double beta0_) {
    assert(beta0_ > 0);
    beta0 = beta0_;
  }
  void set_lambda(const double lambda_) {
    assert(lambda_ > 0);
    lambda = lambda_;
  }
};

#endif  // HYPERSFIXEDNNIG_HPP
