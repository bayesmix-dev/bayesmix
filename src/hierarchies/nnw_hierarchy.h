#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
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
  Hyperparams posterior_hypers;

  // UTILITIES FOR LIKELIHOOD COMPUTATION
  //! Lower factor of the Cholesky decomposition of state.prec
  Eigen::MatrixXd prec_chol;
  //! Determinant of state.prec in logarithmic scale
  double prec_logdet;

  // AUXILIARY TOOLS
  //! Special setter for prec and its utilities
  void set_prec_and_utilities(const Eigen::MatrixXd &prec_);
  //!
  void clear_data() override;
  //!
  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 const Eigen::VectorXd &covariate,
                                 bool add) override;
  //!
  void save_posterior_hypers() override;
  //!
  void initialize_hypers() override;
  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_wishart_update() const;
  //!
  std::shared_ptr<bayesmix::NNWPrior> cast_prior() {
    return std::dynamic_pointer_cast<bayesmix::NNWPrior>(prior);
  }
  //!
  void create_empty_prior() override { prior.reset(new bayesmix::NNWPrior); }

 public:
  void initialize() override;
  //! Returns true if the hierarchy models multivariate data (here, true)
  bool is_multivariate() const override { return true; }
  //!
  void update_hypers(const std::vector<bayesmix::MarginalState::ClusterState>
                         &states) override;

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) const override;
  //! Evaluates the log-marginal distribution of data in a single point
  double marg_lpdf(
      const bool posterior, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) const override;

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
  void sample_given_data(bool update_params = true) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNW;
  }
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
