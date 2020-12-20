#ifndef BAYESMIX_ALGORITHMS_PROBIT_SB_ALGORITHM_HPP_
#define BAYESMIX_ALGORITHMS_PROBIT_SB_ALGORITHM_HPP_

#include <Eigen/Dense>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "conditional_dep_algorithm.hpp"

class ProbitSBAlgorithm : public ConditionalDepAlgorithm {
 protected:
  // ALGORITHM FUNCTIONS
  void print_startup_message() const override;
  void sample_allocations() override;
  void sample_unique_values() override;
  void sample_weights() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~ProbitSBAlgorithm() = default;
  ProbitSBAlgorithm() = default;

  Eigen::MatrixXd eval_lpdf(
      const Eigen::MatrixXd &grid,
      BaseCollector<bayesmix::MarginalState> *const coll) override;

  std::string get_id() const override { return "ProbitSB"; }
};

#endif  // BAYESMIX_ALGORITHMS_PROBIT_SB_ALGORITHM_HPP_
