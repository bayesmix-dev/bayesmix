#ifndef LOAD_HIERARCHIES_HPP
#define LOAD_HIERARCHIES_HPP

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

#endif  // LOAD_HIERARCHIES_HPP
