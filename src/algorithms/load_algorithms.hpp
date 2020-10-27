#ifndef LOAD_ALGORITHMS_HPP
#define LOAD_ALGORITHMS_HPP

#include "../runtime/Factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_Hierarchies() {
  Factory<Algorithm> &factory = Factory<Algorithm>::Instance();
  Builder<Algorithm> Neal2builder = []() { return std::make_shared<Neal2>(); };
  Builder<Algorithm> NNWbuilder = []() { return std::make_shared<Neal8>(); };
  factory.add_builder("N2", Neal2builder);
  factory.add_builder("N8", NNWbuilder);
}

#endif  // LOAD_ALGORITHMS_HPP
