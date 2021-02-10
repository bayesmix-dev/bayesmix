#ifndef BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>

#include "conjugate_hierarchy.h"
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

namespace NNW {
struct State {
  Eigen::VectorXd mean;
  Eigen::MatrixXd prec;
  Eigen::MatrixXd prec_chol;
  double prec_logdet;
};
struct Hyperparams {
  Eigen::VectorXd mean;
  double var_scaling;
  double deg_free;
  Eigen::MatrixXd scale;
  Eigen::MatrixXd scale_inv;
};
}  // namespace NNW

class NNWHierarchy
    : public ConjugateHierarchy<NNWHierarchy, NNW::State, NNW::Hyperparams,
                                bayesmix::NNWPrior> {
 protected:
  unsigned int dim;
  Eigen::VectorXd data_sum;
  Eigen::MatrixXd data_sum_squares;

  // AUXILIARY TOOLS
  //! Special setter for prec and its utilities
  void set_prec_and_utilities(const Eigen::MatrixXd &prec_, NNW::State *out);

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~NNWHierarchy() = default;
  NNWHierarchy() = default;

  bool is_multivariate() const override { return true; }

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) const override;

  double marg_lpdf(
      const NNW::Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) const;

  // SAMPLING FUNCTIONS
  NNW::State draw(const NNW::Hyperparams &params);

  void clear_data();
  void update_hypers(
      const std::vector<bayesmix::MarginalState::ClusterState> &states);

  void initialize_state();
  void initialize_hypers();
  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 const Eigen::VectorXd &covariate, bool add);
  NNW::Hyperparams get_posterior_parameters();

  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::NNW;
  }
};

#endif  // BAYESMIX_HIERARCHIES_NNW_HIERARCHY_H_
