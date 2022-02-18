#ifndef BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_H_
#define BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_H_

#include <functional>
#include <memory>

#include "algorithm_id.pb.h"
#include "base_algorithm.h"
#include "blocked_gibbs_algorithm.h"
#include "neal2_algorithm.h"
#include "neal3_algorithm.h"
#include "neal8_algorithm.h"
#include "split_and_merge_algorithm.h"
#include "src/runtime/factory.h"

//! Loads all available `Algorithm` objects into the appropriate factory, so
//! that they are ready to be chosen and used at runtime.

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using AlgorithmFactory = Factory<bayesmix::AlgorithmId, BaseAlgorithm>;

__attribute__((constructor)) static void load_algorithms() {
  AlgorithmFactory &factory = AlgorithmFactory::Instance();
  // Initialize factory builders
  Builder<BaseAlgorithm> Neal2builder = []() {
    return std::make_shared<Neal2Algorithm>();
  };
  Builder<BaseAlgorithm> Neal3builder = []() {
    return std::make_shared<Neal3Algorithm>();
  };
  Builder<BaseAlgorithm> Neal8builder = []() {
    return std::make_shared<Neal8Algorithm>();
  };
  Builder<BaseAlgorithm> BlockedGibbsbuilder = []() {
    return std::make_shared<BlockedGibbsAlgorithm>();
  };
  Builder<BaseAlgorithm> SplitAndMergebuilder = []() {
    return std::make_shared<SplitAndMergeAlgorithm>();
  };

  factory.add_builder(Neal2Algorithm().get_id(), Neal2builder);
  factory.add_builder(Neal3Algorithm().get_id(), Neal3builder);
  factory.add_builder(Neal8Algorithm().get_id(), Neal8builder);
  factory.add_builder(BlockedGibbsAlgorithm().get_id(), BlockedGibbsbuilder);
  factory.add_builder(SplitAndMergeAlgorithm().get_id(), SplitAndMergebuilder);
}

#endif  // BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_H_
