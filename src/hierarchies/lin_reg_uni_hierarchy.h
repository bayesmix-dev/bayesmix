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

namespace LinRegUni {
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
}  // namespace LinRegUni

class LinRegUniHierarchy
    : public ConjugateHierarchy<LinRegUniHierarchy, LinRegUni::State,
                                LinRegUni::Hyperparams,
                                bayesmix::LinRegUniPrior> {
 public:
 protected:
  unsigned int dim;
  //! Represents pieces of y^t y
  double data_sum_squares;
  //! Represents pieces of X^T X
  Eigen::MatrixXd covar_sum_squares;
  //! Represents pieces of X^t y
  Eigen::VectorXd mixed_prod;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~LinRegUniHierarchy() = default;
  LinRegUniHierarchy() = default;

  bool is_multivariate() const override { return false; }
  bool is_dependent() const override { return true; }
  unsigned int get_dim() const { return dim; }

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(
      const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const override;

  double marg_lpdf(
      const LinRegUni::Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const;

  LinRegUni::State draw(const LinRegUni::Hyperparams &params);

  void clear_data();
  void update_hypers(
      const std::vector<bayesmix::AlgorithmState::ClusterState> &states);

  void initialize_state();
  void initialize_hypers();
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const Eigen::RowVectorXd &covariate, bool add);
  LinRegUni::Hyperparams get_posterior_parameters();

  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::LinRegUni;
  }
};

#endif  // BAYESMIX_HIERARCHIES_LIN_REG_UNI_HIERARCHY_H_
