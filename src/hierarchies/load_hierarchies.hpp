#ifndef BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_HPP_
#define BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_HPP_

#include <functional>
#include <memory>

#include "../runtime/factory.hpp"
#include "base_hierarchy.hpp"
#include "lin_reg_uni_hierarchy.hpp"
#include "nnig_hierarchy.hpp"
#include "nnw_hierarchy.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_hierarchies() {
  Factory<BaseHierarchy> &factory = Factory<BaseHierarchy>::Instance();
  Builder<BaseHierarchy> NNIGbuilder = []() {
    return std::make_shared<NNIGHierarchy>();
  };
  Builder<BaseHierarchy> NNWbuilder = []() {
    return std::make_shared<NNWHierarchy>();
  };
  Builder<BaseHierarchy> LinRegUnibuilder = []() {
    return std::make_shared<LinRegUniHierarchy>();
  };
  factory.add_builder(NNIGHierarchy().get_id(), NNIGbuilder);
  factory.add_builder(NNWHierarchy().get_id(), NNWbuilder);
  factory.add_builder(LinRegUniHierarchy().get_id(), LinRegUnibuilder);
}

#endif  // BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_HPP_
