#ifndef LOAD_HYPERS_HPP
#define LOAD_HYPERS_HPP

#include "../runtime/Factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_hypers() {
  Builder<HypersBase> NNIGFixbuilder = []() {
    return std::make_shared<HypersFixedNNIG>();
  };
  Factory<HypersBase> &factory = Factory<HypersBase>::Instance();
  Builder<HypersBase> NNWFixbuilder = []() {
    return std::make_shared<HypersFixedNNW>();
  };
  factory.add_builder("NNIGFix", NNIGFixbuilder);
  factory.add_builder("NNWFix", NNWFixbuilder);
}

#endif  // LOAD_HYPERS_HPP
