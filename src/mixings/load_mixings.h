#ifndef BAYESMIX_MIXINGS_LOAD_MIXINGS_H_
#define BAYESMIX_MIXINGS_LOAD_MIXINGS_H_

#include <functional>
#include <memory>

#include "base_mixing.h"
#include "dirichlet_mixing.h"
#include "logit_sb_mixing.h"
#include "pityor_mixing.h"
#include "src/runtime/factory.h"
#include "truncated_sb_mixing.h"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using MixingFactory = Factory<bayesmix::MixingId, BaseMixing>;

__attribute__((constructor)) static void load_mixings() {
  MixingFactory &factory = MixingFactory::Instance();
  Builder<BaseMixing> DPbuilder = []() {
    return std::make_shared<DirichletMixing>();
  };
  Builder<BaseMixing> LogSBbuilder = []() {
    return std::make_shared<LogitSBMixing>();
  };
  Builder<BaseMixing> PYbuilder = []() {
    return std::make_shared<PitYorMixing>();
  };
  Builder<BaseMixing> TruncSBbuilder = []() {
    return std::make_shared<TruncatedSBMixing>();
  };
  factory.add_builder(DirichletMixing().get_id(), DPbuilder);
  factory.add_builder(LogitSBMixing().get_id(), LogSBbuilder);
  factory.add_builder(PitYorMixing().get_id(), PYbuilder);
  factory.add_builder(TruncatedSBMixing().get_id(), TruncSBbuilder);
}

#endif  // BAYESMIX_MIXINGS_LOAD_MIXINGS_H_
