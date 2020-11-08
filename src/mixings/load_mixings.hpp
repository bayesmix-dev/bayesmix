#ifndef LOAD_MIXINGS_HPP
#define LOAD_MIXINGS_HPP

#include "../runtime/factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_mixings() {
  Factory<MixingBase> &factory = Factory<MixingBase>::Instance();
  Builder<MixingBase> DPbuilder = []() {
    return std::make_shared<MixingDirichlet>();
  };
  Builder<MixingBase> PYbuilder = []() {
    return std::make_shared<MixingPitYor>();
  };
  factory.add_builder("DP", DPbuilder);
  factory.add_builder("PY", PYbuilder);
}

#endif  // LOAD_MIXINGS_HPP
