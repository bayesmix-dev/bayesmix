#ifndef BAYESMIX_MIXINGS_LOAD_MIXINGS_HPP_
#define BAYESMIX_MIXINGS_LOAD_MIXINGS_HPP_

#include "../runtime/factory.hpp"
#include "mixing_dirichlet.hpp"
#include "mixing_pityor.hpp"
#include "mixing_base.hpp"
#include <functional>
#include <memory>

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

#endif  // BAYESMIX_MIXINGS_LOAD_MIXINGS_HPP_
