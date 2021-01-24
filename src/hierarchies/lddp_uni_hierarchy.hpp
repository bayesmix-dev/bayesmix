#ifndef BAYESMIX_HIERARCHIES_LDDP_UNI_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_LDDP_UNI_HIERARCHY_HPP_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>

#include "../../proto/cpp/hierarchy_prior.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "base_hierarchy.hpp"
#include "dependent_hierarchy.hpp"

class LDDPUniHierarchy : public DependentHierarchy {
 public:
  struct State {
    Eigen::VectorXd regression_coeffs;
    double var;
  };
  struct Hyperparams {
    Eigen::VectorXd mean;
    Eigen::MatrixXd var_scaling;
    double shape;
    double scale;
  };

 protected:
  unsigned int dim;
  //! Represents pieces of y^t y
  double data_sum_squares;
  //! Represents pieces of X^T X
  Eigen::MatrixXd covar_sum_squares;
  //! Represents pieces of X^t y
  Eigen::VectorXd mixed_prod;
  // STATE
  State state;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  // HYPERPRIOR
  std::shared_ptr<bayesmix::LDDUniPrior> prior;

  void clear_data();
  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 bool add);

  // AUXILIARY TOOLS
  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_invgamma_update();

 public:
  void initialize() override;
  //! Returns true if the hierarchy models multivariate data
  bool is_multivariate() const override { return false; }

  void update_hypers(const std::vector<bayesmix::MarginalState::ClusterState>
                         &states) override;

  // DESTRUCTOR AND CONSTRUCTORS
  ~LDDPUniHierarchy() = default;
  LDDPUniHierarchy() = default;

  std::shared_ptr<BaseHierarchy> clone() const override {
    auto out = std::make_shared<LDDPUniHierarchy>(*this);
    out->clear_data();
    return out;
  }

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const override;
  //! Evaluates the log-likelihood of data in the given points
  Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates) const override;
  //! Evaluates the log-marginal distribution of data in a single point
  double marg_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const override;
  //! Evaluates the log-marginal distribution of data in the given points
  Eigen::VectorXd marg_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates) const override;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  void draw() override;
  //! Generates new values for state from the centering posterior distribution
  void sample_given_data() override;
  void sample_given_data(const Eigen::MatrixXd &data) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void set_prior(const google::protobuf::Message &prior_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "LDDPUni"; }
};

#endif  // BAYESMIX_HIERARCHIES_LDDP_UNI_HIERARCHY_HPP_
