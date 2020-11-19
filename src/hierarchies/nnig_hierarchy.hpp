#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_HPP_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <cassert>
#include <memory>

#include "../../proto/cpp/hierarchies.pb.h"
#include "base_hierarchy.hpp"

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
//! state hyperparameters, contained in the Hypers object, are (mu_0, lambda0,
//! alpha0, beta0), all scalar values. Note that this hierarchy is conjugate,
//! thus the marginal and the posterior distribution are available in closed
//! form and Neal's algorithm 2 may be used with it.

class NNIGHierarchy : public BaseHierarchy {
 public:
  struct Hyperparams {
    double mu, alpha, beta, lambda;
  };

 protected:
  // STATE
  double mean, sd;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  // HYPERPRIOR
  bayesmix::NNIGPrior prior;

  // AUXILIARY TOOLS
  //! Raises error if the hypers values are not valid w.r.t. their own domain
  void check_hypers_validity() override {
    assert(hypers->lambda > 0);
    assert(hypers->alpha > 0);
    assert(hypers->beta > 0);
  }

  //! Raises error if the state values are not valid w.r.t. their own domain
  void check_state_validity() override { assert(sd > 0); }

  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_invgamma_update(const Eigen::VectorXd &data,
                                     const double mu0, const double alpha0,
                                     const double beta0, const double lambda0);

 public:
  void check_and_initialize() override;
  //! Returns true if the hierarchy models multivariate data (here, false)
  bool is_multivariate() const override { return false; }

  void update_hypers(
      const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
      unsigned int n) override;

  // DESTRUCTOR AND CONSTRUCTORS
  ~NNIGHierarchy() = default;
  NNIGHierarchy() = default;
  std::shared_ptr<BaseHierarchy> clone() const override {
    return std::make_shared<NNIGHierarchy>(*this);
  }

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(const Eigen::RowVectorXd &datum) const override;
  //! Evaluates the log-likelihood of data in the given points
  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data) const override;
  //! Evaluates the log-marginal distribution of data in a single point
  double marg_lpdf(const Eigen::RowVectorXd &datum) const override;
  //! Evaluates the log-marginal distribution of data in the given points
  Eigen::VectorXd marg_lpdf_grid(const Eigen::MatrixXd &data) const override;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  void draw() override;
  //! Generates new values for state from the centering posterior distribution
  void sample_given_data(const Eigen::MatrixXd &data) override;

  // GETTERS AND SETTERS
  double get_mu0() const { return hypers->mu; }
  double get_alpha0() const { return hypers->alpha; }
  double get_beta0() const { return hypers->beta; }
  double get_lambda0() const { return hypers->lambda; }
  double get_mean() const { return mean; }
  double get_sd() const { return sd; }
  // TODO remove the following 4:
  void set_mu0(const double mu0_) {
    hypers->mu = mu0_;
    mean = hypers->mu;
  }
  void set_alpha0(const double alpha0_) { hypers->alpha = alpha0_; }
  void set_beta0(const double beta0_) { hypers->beta = beta0_; }
  void set_lambda0(const double lambda0_) { hypers->lambda = lambda0_; }

  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  void set_state(const google::protobuf::Message &state_,
                 bool check = true) override;

  void set_prior(const google::protobuf::Message &prior_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "NNIG"; }
};

#endif  // BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_HPP_
