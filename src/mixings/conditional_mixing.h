#ifndef BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_
#define BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_

#include <Eigen/Dense>

#include "base_mixing.h"

class ConditionalMixing : public BaseMixing {
 public:
  //!
  bool is_conditional() const override { return true; }
  //!
  virtual Eigen::VectorXd get_weights(
    const Eigen::MatrixXd &covariates) const = 0;  // TODO default value?
  //! Mass probability for choosing an already existing cluster
  double mass_existing_cluster(const unsigned int n, const bool log,
                               const bool propto,
                               std::shared_ptr<AbstractHierarchy> hier,
                               const Eigen::RowVectorXd &covariate =
                                   Eigen::RowVectorXd(0)) const override {
    throw std::bad_function_call(
      "Conditional mixings do not have mass methods");
    return 0;
  }
  //! Mass probability for choosing a newly created cluster
  double mass_new_cluster(const unsigned int n, const bool log,
                          const bool propto, const unsigned int n_clust,
                          const Eigen::RowVectorXd &covariate =
                              Eigen::RowVectorXd(0)) const override {
    throw std::bad_function_call(
      "Conditional mixings do not have mass methods");
    return 0;
  }
}

#endif  // BAYESMIX_MIXINGS_CONDITIONAL_MIXING_H_
