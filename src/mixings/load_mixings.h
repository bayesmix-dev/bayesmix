#ifndef BAYESMIX_MIXINGS_LOAD_MIXINGS_H_
#define BAYESMIX_MIXINGS_LOAD_MIXINGS_H_

#include <functional>
#include <memory>

#include "base_mixing.hpp"
#include "dirichlet_mixing.hpp"
#include "pityor_mixing.hpp"
#include "src/runtime/factory.hpp"

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
  factory.add_builder(DirichletMixing().get_id(), DPbuilder);
  factory.add_builder(PitYorMixing().get_id(), PYbuilder);
}

#endif  // BAYESMIX_MIXINGS_LOAD_MIXINGS_H_