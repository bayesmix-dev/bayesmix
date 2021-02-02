#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>

#include "base_hierarchy.h"
#include "hierarchy_prior.pb.h"
#include "marginal_state.pb.h"

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
    Eigen::VectorXd mean;
    double var_scaling;
    double deg_free;
    Eigen::MatrixXd scale;
    Eigen::MatrixXd scale_inv;
  };

 protected:
  unsigned int dim;
  Eigen::VectorXd data_sum;
  Eigen::MatrixXd data_sum_squares;
  // STATE
  State state;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  // HYPERPRIOR
  std::shared_ptr<bayesmix::NNWPrior> prior;

  // UTILITIES FOR LIKELIHOOD COMPUTATION
  //! Lower factor of the Cholesky decomposition of prec
  Eigen::MatrixXd prec_chol;
  //! Determinant of prec in logarithmic scale
  double prec_logdet;

  // AUXILIARY TOOLS
  //! Special setter for prec and its utilities
  void set_prec_and_utilities(const Eigen::MatrixXd &prec_);

  void clear_data() override;

  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 bool add) override;

  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_wishart_update();

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const override;
  //! Evaluates the log-marginal distribution of data in a single point
  template<bool posterior>
  double marg_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const override {
    Hyperparams params = posterior ? normal_wishart_update() : *hypers;
    // Compute dof and scale of marginal distribution
    double nu_n = 2 * params.deg_free - dim + 1;
    Eigen::MatrixXd sigma_n = params.scale_inv *
                              (params.deg_free - 0.5 * (dim - 1)) *
                              params.var_scaling / (params.var_scaling + 1);

    // TODO: chec if this is optimized as our bayesmix::multi_normal_prec_lpdf
    return stan::math::multi_student_t_lpdf(datum, nu_n, hypers->mean, sigma_n);
  }

 public:
  void initialize() override;
  //! Returns true if the hierarchy models multivariate data (here, true)
  bool is_multivariate() const override { return true; }
  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return false; }

  void update_hypers(const std::vector<bayesmix::MarginalState::ClusterState>
                         &states) override;

  // DESTRUCTOR AND CONSTRUCTORS
  ~NNWHierarchy() = default;
  NNWHierarchy() = default;
  std::shared_ptr<BaseHierarchy> clone() const override {
    auto out = std::make_shared<NNWHierarchy>(*this);
    out->clear_data();
    return out;
  }

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  void draw() override;
  //! Generates new values for state from the centering posterior distribution
  void sample_given_data() override;
  void sample_given_data(const Eigen::MatrixXd &data) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void set_prior(const google::protobuf::Message &prior_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "NNW"; }
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
