#ifndef BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_HPP_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>

#include "base_hierarchy.hpp"
#include "dependent_hierarchy.hpp"
#include "hierarchy_prior.pb.h"
#include "marginal_state.pb.h"

class LinRegUniHierarchy : public DependentHierarchy {
 public:
  struct State {
    Eigen::VectorXd regression_coeffs;
    double var;
  };
  struct Hyperparams {
    Eigen::VectorXd mean;
    Eigen::MatrixXd var_scaling;
    Eigen::MatrixXd var_scaling_inv;
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
  std::shared_ptr<bayesmix::LinRegUniPrior> prior;

  void clear_data();
  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 const Eigen::VectorXd &covariate, bool add);

  // AUXILIARY TOOLS
  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_invgamma_update();

  void sample_given_data(const Eigen::MatrixXd &data,
                         const Eigen::MatrixXd &covariates) override;

 public:
  void initialize() override;
  //! Returns true if the hierarchy models multivariate data
  bool is_multivariate() const override { return false; }

  void update_hypers(const std::vector<bayesmix::MarginalState::ClusterState>
                         &states) override;

  // DESTRUCTOR AND CONSTRUCTORS
  ~LinRegUniHierarchy() = default;
  LinRegUniHierarchy() = default;

  std::shared_ptr<BaseHierarchy> clone() const override {
    auto out = std::make_shared<LinRegUniHierarchy>(*this);
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

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void set_prior(const google::protobuf::Message &prior_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "LinRegUni"; }
};

#endif  // BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_HPP_
