#ifndef BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_
#define BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_

#include <Eigen/Dense>

#include "base_mixing.h"

template <class Derived, typename State, typename Prior>
class ConditionalMixing
    : public BaseMixing<Derived, State, Prior> {
 public:
  ~ConditionalMixing() = default;
  ConditionalMixing() = default;
  //!
  bool is_conditional() const override { return true; }
  //!
  virtual Eigen::VectorXd get_weights(
      const bool log, const bool propto,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const = 0;
};

#endif  // BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_
