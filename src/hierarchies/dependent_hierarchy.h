#ifndef BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_H_

#include "base_hierarchy.h"

class DependentHierarchy : public BaseHierarchy {
 protected:
  unsigned int dim;

 public:
  //! Returns true if the hierarchy has covariates i.e. is a dependent model
  bool is_dependent() const override { return true; }

  // DESTRUCTOR AND CONSTRUCTORS
  ~DependentHierarchy() = default;
  DependentHierarchy() = default;

  // GETTERS AND SETTERS
  unsigned int get_dim() const { return dim; }
};

#endif  // BAYESMIX_HIERARCHIES_DEPENDENT_HIERARCHY_H_
