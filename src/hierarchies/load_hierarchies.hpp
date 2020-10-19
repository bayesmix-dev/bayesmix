#ifndef FACTORY_LOAD_HPP
#define FACTORY_LOAD_HPP

#include "../runtime/Factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_Hierarchies() {
  Builder<HierarchyBase> NNIGbuilder = []() {
    return std::make_shared<HierarchyNNIG>();
  };
  Factory<HierarchyBase> &factory = Factory<HierarchyBase>::Instance();
  Builder<HierarchyBase> NNWbuilder = []() {
    return std::make_shared<HierarchyNNW>();
  };
  factory.add_builder("NNIG", NNIGbuilder);
  factory.add_builder("NNW", NNWbuilder);
}

#endif  // FACTORY_LOAD_HPP
