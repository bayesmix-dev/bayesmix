#ifndef BAYESMIX_ALGORITHMS_PROBIT_SB_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_PROBIT_SB_ALGORITHM_HPP_

#include <Eigen/Dense>
#include <memory>

#include "../hierarchies/base_hierarchy.hpp"
#include "conditional_dependent_algorithm.hpp"

class ProbitSBAlgorithm : public ConditionalDependentAlgorithm {
 protected:
  // ALGORITHM FUNCTIONS
  void print_startup_message() const override;
  void sample_allocations() override;
  void sample_unique_values() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~ProbitSBAlgorithm() = default;
  ProbitSBAlgorithm() = default;

  std::string get_id() const override { return "ProbitSB"; }
};

#endif  // BAYESMIX_ALGORITHMS_PROBIT_SB_ALGORITHM_HPP_
