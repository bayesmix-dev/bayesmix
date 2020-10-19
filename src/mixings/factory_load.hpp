#ifndef FACTORY_LOAD_HPP
#define FACTORY_LOAD_HPP

#include "../runtime/Factory.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_Mixing() {
  Builder<BaseMixing> DPbuilder = []() {
    return std::make_shared<DirichletMixing>();
  };
  Factory<BaseMixing> &factory = Factory<BaseMixing>::Instance();
  Builder<BaseMixing> PYbuilder = []() {
    return std::make_shared<PitYorMixing>();
  };
  factory.add_builder("DP", DPbuilder);
  factory.add_builder("PY", PYbuilder);
}

#endif  // FACTORY_LOAD_HPP
