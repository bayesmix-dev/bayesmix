#ifndef BAYESMIX_MIXINGS_LOAD_MIXINGS_H_
#define BAYESMIX_MIXINGS_LOAD_MIXINGS_H_

#include <functional>
#include <memory>

#include "abstract_mixing.h"
#include "dirichlet_mixing.h"
#include "logit_sb_mixing.h"
#include "mixture_finite_mixing.h"
#include "pityor_mixing.h"
#include "src/runtime/factory.h"
#include "truncated_sb_mixing.h"

//! Loads all available `Mixing` objects into the appropriate factory, so that
//! they are ready to be chosen and used at runtime.

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using MixingFactory = Factory<bayesmix::MixingId, AbstractMixing>;

__attribute__((constructor)) static void load_mixings() {
  MixingFactory &factory = MixingFactory::Instance();
  // Initialize factory builders
  Builder<AbstractMixing> DPbuilder = []() {
    return std::make_shared<DirichletMixing>();
  };
  Builder<AbstractMixing> LogSBbuilder = []() {
    return std::make_shared<LogitSBMixing>();
  };
  Builder<AbstractMixing> MFMbuilder = []() {
    return std::make_shared<MixtureFiniteMixing>();
  };
  Builder<AbstractMixing> PYbuilder = []() {
    return std::make_shared<PitYorMixing>();
  };
  Builder<AbstractMixing> TruncSBbuilder = []() {
    return std::make_shared<TruncatedSBMixing>();
  };

  factory.add_builder(DirichletMixing().get_id(), DPbuilder);
  factory.add_builder(LogitSBMixing().get_id(), LogSBbuilder);
  factory.add_builder(MixtureFiniteMixing().get_id(), MFMbuilder);
  factory.add_builder(PitYorMixing().get_id(), PYbuilder);
  factory.add_builder(TruncatedSBMixing().get_id(), TruncSBbuilder);
}

#endif  // BAYESMIX_MIXINGS_LOAD_MIXINGS_H_
