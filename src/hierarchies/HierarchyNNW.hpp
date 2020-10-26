#ifndef HIERARCHYNNW_HPP
#define HIERARCHYNNW_HPP

#include <Eigen/Dense>

#include "HierarchyBase.hpp"
//#include <stan/math/prim.hpp>  // TODO

//! Normal Normal-Wishart hierarchy for multivariate data.

//! This class represents a hierarchy, i.e. a cluster, whose multivariate data
//! are distributed according to a multinomial normal likelihood, the
//! parameters of which have a Normal-Wishart centering distribution. That is:
//!           phi = (mu,tau)   (state);
//! f(x_i|mu,tau) = N(mu,tau)  (data likelihood);
//!      (mu,tau) ~ G          (unique values distribution);
//!             G ~ MM         (mixture model);
//!            G0 = N-W        (centering distribution).
//! state[0] = mu is called location, and state[1] = tau is called precision.
//! The state's hyperparameters, contained in the Hypers object, are (mu0,
//! lambda, tau0, nu), which are respectively vector, scalar, matrix, and
//! scalar. Note that this hierarchy is conjugate, thus the marginal and the
//! posterior distribution are available in closed form and Neal's algorithm 2
//! may be used with it.

class HierarchyNNW : public HierarchyBase {
 protected:
  using HierarchyBase::state;
  using EigenRowVec = Eigen::Matrix<double, 1, Eigen::Dynamic>;

  // HYPERPARAMETERS
  EigenRowVec mu0;
  double lambda;
  Eigen::MatrixXd tau0, tau0_inv;
  double nu;

  // UTILITIES FOR LIKELIHOOD COMPUTATION
  //! Lower factor object of the Cholesky decomposition of tau
  Eigen::LLT<Eigen::MatrixXd> tau_chol_factor;
  //! Matrix-form evaluation of tau_chol_factor
  Eigen::MatrixXd tau_chol_factor_eval;
  //! Determinant of tau in logarithmic scale
  double tau_log_det;

  // AUXILIARY TOOLS
  //! Raises error if the hypers values are not valid w.r.t. their own domain
  void check_hypers_validity() override;
  //! Raises error if the state values are not valid w.r.t. their own domain
  void check_state_validity() override;
  //! Special setter tau and its utilities
  void set_tau_and_utilities(const Eigen::MatrixXd &tau);

  //! Returns updated values of the prior hyperparameters via their posterior
  std::vector<Eigen::MatrixXd> normal_wishart_update(
      const Eigen::MatrixXd &data, const EigenRowVec &mu0, const double lambda,
      const Eigen::MatrixXd &tau0_inv, const double nu);

 public:
  //! Returns true if the hierarchy models multivariate data (here, true)
  bool is_multivariate() const override { return true; }

  // DESTRUCTOR AND CONSTRUCTORS
  ~HierarchyNNW() = default;
  HierarchyNNW() {
    unsigned int dim = get_mu0().size();
    state.push_back(get_mu0());
    set_tau_and_utilities(get_lambda() * Eigen::MatrixXd::Identity(dim, dim));
  }

  // EVALUATION FUNCTIONS
  //! Evaluates the likelihood of data in the given points
  Eigen::VectorXd like(const Eigen::MatrixXd &data) override;

  //! Evaluates the log-likelihood of data in the given points
  Eigen::VectorXd lpdf(const Eigen::MatrixXd &data) override;

  //! Evaluates the marginal distribution of data in the given points
  Eigen::VectorXd eval_marg(const Eigen::MatrixXd &data) override;

  //! Evaluates the log-marginal distribution of data in the given points
  Eigen::VectorXd marg_lpdf(const Eigen::MatrixXd &data) override;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  void draw() override;
  //! Generates new values for state from the centering posterior distribution
  void sample_given_data(const Eigen::MatrixXd &data) override;

  // GETTERS AND SETTERS
  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  void set_state(const std::vector<Eigen::MatrixXd> &state_,
                 bool check = true) override {
    state[0] = state_[0];
    set_tau_and_utilities(state_[1]);
    if (check) {
      check_state_validity();
    }
  }

  EigenRowVec get_mu0() const { return mu0; }

  double get_lambda() const { return lambda; }

  Eigen::MatrixXd get_tau0() const { return tau0; }

  Eigen::MatrixXd get_tau0_inv() const { return tau0_inv; }

  double get_nu() const { return nu; }

  void set_mu0(const EigenRowVec &mu0_) {
    assert(mu0_.size() == mu0.size());
    mu0 = mu0_;
  }

  void set_lambda(const double lambda_) {
    assert(lambda_ > 0);
    lambda = lambda_;
  }

  void set_tau0(const Eigen::MatrixXd &tau0_);

  void set_nu(const double nu_) {
    assert(nu_ > mu0.size() - 1);
    nu = nu_;
  }

  void print_id() const override { std::cout << "NNW" << std::endl; }  // TODO
};

#endif  // HIERARCHYNNW_HPP
