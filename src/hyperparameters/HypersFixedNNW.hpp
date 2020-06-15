#ifndef HYPERSFIXEDNNW_HPP
#define HYPERSFIXEDNNW_HPP

#include <Eigen/Dense>
#include <cassert>

//! Class that represents fixed hyperparameters for an NNW hierarchy.

//! That is, it represents hyperparameters without a prior distribution. It can
//! be used as a template argument for the multivariate template class,
//! HierarchyNNW. All constructors and setters have validity checks for the
//! inserted values.

class HypersFixedNNW {
 protected:
  using EigenRowVec = Eigen::Matrix<double, 1, Eigen::Dynamic>;

  // HYPERPARAMETERS
  EigenRowVec mu0;
  double lambda;
  Eigen::MatrixXd tau0;
  double nu;

  //! Raises error if the hypers values are not valid w.r.t. their own domain
  void check_hypers_validity() {
    unsigned int dim = mu0.size();
    assert(lambda > 0);
    assert(dim == tau0.rows());
    assert(nu > dim - 1);

    // Check if tau0 is a square symmetric positive semidefinite matrix
    assert(tau0.rows() == tau0.cols());
    assert(tau0.isApprox(tau0.transpose()));
    Eigen::LLT<Eigen::MatrixXd> llt(tau0);
    assert(llt.info() != Eigen::NumericalIssue);
  }

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~HypersFixedNNW() = default;
  HypersFixedNNW() = default;
  HypersFixedNNW(const EigenRowVec &mu0_, const double lambda_,
                 const Eigen::MatrixXd &tau0_, const double nu_)
      : mu0(mu0_), lambda(lambda_), tau0(tau0_), nu(nu_) {
    check_hypers_validity();
  }

  // GETTERS AND SETTERS
  EigenRowVec get_mu0() const { return mu0; }
  double get_lambda() const { return lambda; }
  Eigen::MatrixXd get_tau0() const { return tau0; }
  double get_nu() const { return nu; }
  void set_mu0(const EigenRowVec &mu0_) {
    assert(mu0_.size() == mu0.size());
    mu0 = mu0_;
  }
  void set_lambda(const double lambda_) {
    assert(lambda_ > 0);
    lambda = lambda_;
  }
  void set_tau0(const Eigen::MatrixXd &tau0_) {
    // Check if tau0 is a square symmetric positive semidefinite matrix
    assert(tau0_.rows() == tau0_.cols());
    assert(mu0.size() == tau0_.rows());
    assert(tau0.isApprox(tau0.transpose()));
    Eigen::LLT<Eigen::MatrixXd> llt(tau0);
    assert(llt.info() != Eigen::NumericalIssue);
    tau0 = tau0_;
  }
  void set_nu(const double nu_) {
    assert(nu_ > mu0.size() - 1);
    nu = nu_;
  }
};

#endif  // HYPERSFIXEDNNW_HPP
