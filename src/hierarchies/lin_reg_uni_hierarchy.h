#ifndef BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_state.pb.h"
#include "conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

//! Linear regression hierarchy for univariate data.

//! This class implements a dependent hierarchy which represents the classical
//! univariate Bayesian linear regression model, i.e.:
//!    y_i | \beta, x_i, \sigma^2 \sim N(\beta^T x_i, sigma^2)
//!              \beta | \sigma^2 \sim N(\mu, sigma^2 Lambda^{-1})
//!                      \sigma^2 \sim InvGamma(a, b)
//!
//! The state consists of the `regression_coeffs` \beta, and the `var` sigma^2.
//! Lambda is called the variance-scaling factor. For more information, please
//! refer to parent classes: `AbstractHierarchy`, `BaseHierarchy`, and
//! `ConjugateHierarchy`.

namespace LinRegUni {
//! Custom container for State values
struct State {
  Eigen::VectorXd regression_coeffs;
  double var;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  Eigen::VectorXd mean;
  Eigen::MatrixXd var_scaling;
  Eigen::MatrixXd var_scaling_inv;
  double shape;
  double scale;
};
}  // namespace LinRegUni

class LinRegUniHierarchy
    : public ConjugateHierarchy<LinRegUniHierarchy, LinRegUni::State,
                                LinRegUni::Hyperparams,
                                bayesmix::LinRegUniPrior> {
 public:
  LinRegUniHierarchy() = default;
  ~LinRegUniHierarchy() = default;

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double marg_lpdf(const LinRegUni::Hyperparams &params,
                   const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const;

  void initialize_state();

  void initialize_hypers();

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  LinRegUni::State draw(const LinRegUni::Hyperparams &params);

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param covariate  Covariate vector associated to datum
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate,
                                 bool add);

  //! Removes every data point from this cluster
  void clear_data();

  bool is_multivariate() const override { return false; }

  unsigned int get_dim() const { return dim; }

  //! Computes and return posterior hypers given data currently in this cluster
  LinRegUni::Hyperparams get_posterior_parameters();

  void set_state_from_proto(const google::protobuf::Message &state_) override;

  void write_state_to_proto(google::protobuf::Message *out) const override;

  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LinRegUni;
  }

  const bool IS_DEPENDENT = true;

 protected:
  //! Private version of get_like_lpdf()
  double like_lpdf(const Eigen::RowVectorXd &datum,
                   const Eigen::RowVectorXd &covariate) const override;

  //! Dimension of the coefficients vector
  unsigned int dim;

  //! Represents pieces of y^t y
  double data_sum_squares;

  //! Represents pieces of X^T X
  Eigen::MatrixXd covar_sum_squares;

  //! Represents pieces of X^t y
  Eigen::VectorXd mixed_prod;
};

#endif  // BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
