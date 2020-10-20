#ifndef LOAD_MIXINGS_HPP
#define LOAD_MIXINGS_HPP

#include "../runtime/Factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_mixings() {
  Factory<BaseMixing> &factory = Factory<BaseMixing>::Instance();
  Builder<BaseMixing> DPbuilder = []() {
    return std::make_shared<DirichletMixing>();
  };
  Builder<BaseMixing> PYbuilder = []() {
    return std::make_shared<PitYorMixing>();
  };
  factory.add_builder("DP", DPbuilder);
  factory.add_builder("PY", PYbuilder);
}

#endif  // LOAD_MIXINGS_HPP
