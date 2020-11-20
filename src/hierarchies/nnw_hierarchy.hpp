#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_HPP_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>

#include "../../proto/cpp/hierarchies.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "base_hierarchy.hpp"

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
//! lambda0, tau0, nu0), which are respectively vector, scalar, matrix, and
//! scalar. Note that this hierarchy is conjugate, thus the marginal and the
//! posterior distribution are available in closed form and Neal's algorithm 2
//! may be used with it.

class NNWHierarchy : public BaseHierarchy {
 public:
  struct State {
    Eigen::VectorXd mean;
    Eigen::MatrixXd prec;
  };
  struct Hyperparams {
    Eigen::RowVectorXd mu;
    double lambda;
    Eigen::MatrixXd tau;
    double nu;
  };

 protected:
  // STATE
  State state;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  Eigen::MatrixXd tau0_inv;
  // HYPERPRIOR
  bayesmix::NNWPrior prior;

  // UTILITIES FOR LIKELIHOOD COMPUTATION
  //! Lower factor object of the Cholesky decomposition of prec
  Eigen::LLT<Eigen::MatrixXd> prec_chol_factor;
  //! Matrix-form evaluation of prec_chol_factor
  Eigen::MatrixXd prec_chol_factor_eval;  // TODO do we need both?
  //! Determinant of prec in logarithmic scale
  double prec_logdet;

  // AUXILIARY TOOLS
  //! Raises error if the state values are not valid w.r.t. their own domain
  void check_state_validity() override;
  //! Special setter for prec and its utilities
  void set_prec_and_utilities(const Eigen::MatrixXd &prec_);

  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_wishart_update(const Eigen::MatrixXd &data,
                                    const Eigen::RowVectorXd &mu0,
                                    const double lambda0,
                                    const Eigen::MatrixXd &tau0_inv,
                                    const double nu0);

 public:
  void initialize() override;
  //! Returns true if the hierarchy models multivariate data (here, true)
  bool is_multivariate() const override { return true; }

  void update_hypers(
      const std::vector<bayesmix::MarginalState::ClusterVal> &states) override;

  // DESTRUCTOR AND CONSTRUCTORS
  ~NNWHierarchy() = default;
  NNWHierarchy() = default;
  std::shared_ptr<BaseHierarchy> clone() const override {
    return std::make_shared<NNWHierarchy>(*this);
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
  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  Eigen::MatrixXd get_tau0_inv() const { return tau0_inv; }

  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  void set_state_from_proto(const google::protobuf::Message &state_,
                            bool check = true) override;

  void set_prior(const google::protobuf::Message &prior_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "NNW"; }
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_HPP_
