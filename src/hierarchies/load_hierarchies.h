#ifndef BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_H_
#define BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_H_

#include <functional>
#include <memory>

#include "abstract_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "lapnig_hierarchy.h"
#include "lin_reg_uni_hierarchy.h"
#include "mfa_hierarchy.h"
#include "nnig_hierarchy.h"
#include "nnw_hierarchy.h"
#include "src/runtime/factory.h"

//! Loads all available `Hierarchy` objects into the appropriate factory, so
//! that they are ready to be chosen and used at runtime.

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using HierarchyFactory = Factory<bayesmix::HierarchyId, AbstractHierarchy>;

__attribute__((constructor)) static void load_hierarchies() {
  HierarchyFactory &factory = HierarchyFactory::Instance();
  // Initialize factory builders
  Builder<AbstractHierarchy> NNIGbuilder = []() {
    return std::make_shared<NNIGHierarchy>();
  };
  Builder<AbstractHierarchy> NNWbuilder = []() {
    return std::make_shared<NNWHierarchy>();
  };
  Builder<AbstractHierarchy> LinRegUnibuilder = []() {
    return std::make_shared<LinRegUniHierarchy>();
  };
  Builder<AbstractHierarchy> MFAbuilder = []() {
    return std::make_shared<MFAHierarchy>();
    Builder<AbstractHierarchy> LapNIGbuilder = []() {
      return std::make_shared<LapNIGHierarchy>();
    };

    factory.add_builder(LapNIGHierarchy().get_id(), LapNIGbuilder);
    factory.add_builder(LinRegUniHierarchy().get_id(), LinRegUnibuilder);
    factory.add_builder(NNIGHierarchy().get_id(), NNIGbuilder);
    factory.add_builder(NNWHierarchy().get_id(), NNWbuilder);
    factory.add_builder(LinRegUniHierarchy().get_id(), LinRegUnibuilder);
    factory.add_builder(MFAHierarchy().get_id(), MFAbuilder);
  }

#endif  // BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_H_
