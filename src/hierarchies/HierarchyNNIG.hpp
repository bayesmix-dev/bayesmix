#ifndef HIERARCHYNNIG_HPP
#define HIERARCHYNNIG_HPP

#include <google/protobuf/stubs/casts.h>

#include "../../proto/cpp/ls_state.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "../utils/rng.hpp"
#include "HierarchyBase.hpp"

//! Normal Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchy, i.e. a cluster, whose univariate data
//! are distributed according to a normal likelihood, the parameters of which
//! have a Normal-InverseGamma centering distribution. That is:
//!           phi = (mu,sig)     (state);
//! f(x_i|mu,sig) = N(mu,sig^2)  (data likelihood);
//!    (mu,sig^2) ~ G            (unique values distribution);
//!             G ~ MM           (mixture model);
//!            G0 = N-IG         (centering distribution).
//! state[0] = mu is called location, and state[1] = sig is called scale. The
//! state's hyperparameters, contained in the Hypers object, are (mu_0, lambda,
//! alpha, beta), all scalar values. Note that this hierarchy is conjugate,
//! thus the marginal and the posterior distribution are available in closed
//! form and Neal's algorithm 2 may be used with it.

class HierarchyNNIG : public HierarchyBase {
 protected:
  // state
  double mean;
  double std = 1.0;

  // HYPERPARAMETERS
  double mu0, lambda, alpha0, beta0;

  // AUXILIARY TOOLS
  //! Raises error if the hypers values are not valid w.r.t. their own domain
  void check_hypers_validity() override {
    assert(lambda > 0);
    assert(alpha0 > 0);
    assert(beta0 > 0);
  }

  //! Raises error if the state values are not valid w.r.t. their own domain
  void check_state_validity() override;

  //! Returns updated values of the prior hyperparameters via their posterior
  std::vector<double> normal_gamma_update(const Eigen::VectorXd &data,
                                          const double mu0,
                                          const double alpha0,
                                          const double beta0,
                                          const double lambda);

 public:
  //! Returns true if the hierarchy models multivariate data (here, false)
  bool is_multivariate() const override { return false; }

  // DESTRUCTOR AND CONSTRUCTORS
  ~HierarchyNNIG() = default;
  HierarchyNNIG() = default;
  std::shared_ptr<HierarchyBase> clone() const override {
    return std::make_shared<HierarchyNNIG>(*this);
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
  double get_mu0() const { return mu0; }
  double get_alpha0() const { return alpha0; }
  double get_beta0() const { return beta0; }
  double get_lambda() const { return lambda; }
  double get_mean() const { return mean; }
  void set_mu0(const double mu0_) { mu0 = mu0_; mean = mu0; }
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

  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  void set_state(google::protobuf::Message *curr, bool check = true) override;

  void get_state_as_proto(google::protobuf::Message *out);

  void print_id() const override { std::cout << "NNIG" << std::endl; }  // TODO
};

#endif  // HIERARCHYNNIG_HPP
