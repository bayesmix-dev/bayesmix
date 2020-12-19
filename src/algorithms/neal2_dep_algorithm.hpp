#ifndef BAYESMIX_ALGORITHMS_NEAL2_DEP_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_NEAL2_DEP_ALGORITHM_HPP_

#include <Eigen/Dense>
#include <memory>

#include "../hierarchies/base_hierarchy.hpp"
#include "marginal_dependent_algorithm.hpp"

class Neal2DepAlgorithm : public MarginalDependentAlgorithm {
 protected:
  // AUXILIARY TOOLS
  //! Computes marginal contribution of a given iteration & cluster
  Eigen::VectorXd lpdf_marginal_component(
      std::shared_ptr<BaseHierarchy> temp_hier,
      const Eigen::MatrixXd &grid) override;

  // ALGORITHM FUNCTIONS
  void print_startup_message() const override;
  void sample_allocations() override;
  void sample_unique_values() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~Neal2DepAlgorithm() = default;
  Neal2DepAlgorithm() = default;

  std::string get_id() const override { return "Neal2dep"; }
};

#endif  // BAYESMIX_ALGORITHMS_NEAL2_DEP_ALGORITHM_HPP_
