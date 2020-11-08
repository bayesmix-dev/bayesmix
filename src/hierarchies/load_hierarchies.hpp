#ifndef BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_HPP_
#define BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_HPP_

#include "../runtime/factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_hierarchies() {
  Factory<HierarchyBase> &factory = Factory<HierarchyBase>::Instance();
  Builder<HierarchyBase> NNIGbuilder = []() {
    return std::make_shared<HierarchyNNIG>();
  };
  Builder<HierarchyBase> NNWbuilder = []() {
    return std::make_shared<HierarchyNNW>();
  };
  factory.add_builder("NNIG", NNIGbuilder);
  factory.add_builder("NNW", NNWbuilder);
}

#endif  // BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_HPP_
