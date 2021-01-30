#ifndef BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_

#include <google/protobuf/message.h>

#include <Eigen/Dense>

#include "../collectors/base_collector.hpp"
#include "../hierarchies/base_hierarchy.hpp"
#include "../hierarchies/dependent_hierarchy.hpp"
#include "base_algorithm.hpp"
#include "marginal_state.pb.h"

class MarginalAlgorithm : public BaseAlgorithm {
 protected:
  bayesmix::MarginalState curr_state;

  //! Computes marginal contribution of a given iteration & cluster
  virtual Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<BaseHierarchy> temp_hier,
      const Eigen::MatrixXd &grid) = 0;

  virtual Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<DependentHierarchy> temp_hier,
      const Eigen::MatrixXd &grid, const Eigen::MatrixXd &covariates) = 0;

 public:
  ~MarginalAlgorithm() = default;
  MarginalAlgorithm() = default;
  Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                            BaseCollector *coll) override;
  Eigen::MatrixXd eval_lpdf(const Eigen::MatrixXd &grid,
                            const Eigen::MatrixXd &covariates,
                            BaseCollector *coll) override;

  Eigen::VectorXd lpdf_from_state(const Eigen::MatrixXd &grid);
  Eigen::VectorXd lpdf_from_state(const Eigen::MatrixXd &grid,
                                  const Eigen::MatrixXd &covariates);

  bool update_state_from_collector(BaseCollector *coll);
};

#endif  // BAYESMIX_ALGORITHMS_MARGINAL_ALGORITHM_HPP_
