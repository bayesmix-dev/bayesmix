#ifndef BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_H_
#define BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_H_

#include <functional>
#include <memory>

#include "src/runtime/factory.hpp"
#include "base_algorithm.hpp"
#include "neal2_algorithm.hpp"
#include "neal8_algorithm.hpp"

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_algorithms() {
  Factory<BaseAlgorithm> &factory = Factory<BaseAlgorithm>::Instance();
  Builder<BaseAlgorithm> Neal2builder = []() {
    return std::make_shared<Neal2Algorithm>();
  };
  Builder<BaseAlgorithm> Neal8builder = []() {
    return std::make_shared<Neal8Algorithm>();
  };
  factory.add_builder(Neal2Algorithm().get_id(), Neal2builder);
  factory.add_builder(Neal8Algorithm().get_id(), Neal8builder);
}

#endif  // BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_H_
