#ifndef BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <memory>

#include "base_hierarchy.h"
#include "dependent_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"
#include "marginal_state.pb.h"

namespace LinReg {
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
}  // namespace LinReg

class LinRegUniHierarchy
    : public DependentHierarchy<LinRegUniHierarchy, LinReg::State,
                                LinReg::Hyperparams,
                                bayesmix::LinRegUniPrior> {
 public:

 protected:
  //! Represents pieces of y^t y
  double data_sum_squares;
  //! Represents pieces of X^T X
  Eigen::MatrixXd covar_sum_squares;
  //! Represents pieces of X^t y
  Eigen::VectorXd mixed_prod;

  std::shared_ptr<bayesmix::LinRegUniPrior> cast_prior() {
    return std::dynamic_pointer_cast<bayesmix::LinRegUniPrior>(prior);
  }

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~LinRegUniHierarchy() = default;
  LinRegUniHierarchy() = default;

  bool is_multivariate() const override { return false; }

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0)) override;

  double marg_lpdf(
      const LinReg::Hyperparams & params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::VectorXd(0));

  LinReg::State draw(const LinReg::Hyperparams & params);

  void clear_data();
  void update_hypers(const std::vector<bayesmix::MarginalState::ClusterState>
                         &states) override;

  void initialize_state();
  void initialize_hypers();
  void update_summary_statistics(const Eigen::VectorXd &datum,
                                 const Eigen::VectorXd &covariate, bool add);
  LinReg::Hyperparams get_posterior_parameters();

  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LinRegUni;
  }
};

#endif  // BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
