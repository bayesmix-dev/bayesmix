#ifndef BAYESMIX_MIXINGS_DEPENDENT_MIXING_H_
#define BAYESMIX_MIXINGS_DEPENDENT_MIXING_H_

#include "base_mixing.h"

class DependentMixing : public BaseMixing {
 protected:
  unsigned int dim;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~DependentMixing() = default;
  DependentMixing() = default;

  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return true; }

  // GETTERS AND SETTERS
  unsigned int get_dim() const { return dim; }
};

#endif  // BAYESMIX_MIXINGS_DEPENDENT_MIXING_H_
