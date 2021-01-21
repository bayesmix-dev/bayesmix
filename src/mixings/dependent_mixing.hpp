#ifndef BAYESMIX_MIXINGS_DEPENDENT_MIXING_HPP_
#define BAYESMIX_MIXINGS_DEPENDENT_MIXING_HPP_

#include "base_mixing.hpp"

class DependentMixing : public BaseMixing {
 protected:
  unsigned int dim;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~DependentMixing() = default;
  DependentMixing() = default;
  //! Returns true if the mixing has covariates i.e. is a dependent model
  bool is_dependent() const override { return true; }
  // GETTERS AND SETTERS
  unsigned int get_parameters_dim() { return dim; }
  void set_parameters_dim(const unsigned int dim_) { dim = dim_; }
};

#endif  // BAYESMIX_MIXINGS_DEPENDENT_MIXING_HPP_
