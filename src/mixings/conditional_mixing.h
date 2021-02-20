#ifndef BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_
#define BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_

#include <Eigen/Dense>

#include "base_mixing.h"

class ConditionalMixing : public BaseMixing {
 public:
  ~ConditionalMixing() = default;
  ConditionalMixing() = default;
  //!
  bool is_conditional() const override { return true; }
  //!
  virtual Eigen::VectorXd get_weights(
    const Eigen::VectorXd &covariate) const = 0;  // TODO default value?
}

#endif  // BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_
