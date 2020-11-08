#ifndef BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_HPP_
#define BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_HPP_

#include "../runtime/factory.hpp"
#include "algorithm_base.hpp"
#include "algorithm_neal2.hpp"
#include "algorithm_neal8.hpp"
#include <functional>
#include <memory>

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor)) static void load_algorithms() {
  Factory<AlgorithmBase> &factory = Factory<AlgorithmBase>::Instance();
  Builder<AlgorithmBase> Neal2builder = []() {
  	return std::make_shared<AlgorithmNeal2>();
  };
  Builder<AlgorithmBase> Neal8builder = []() {
  	return std::make_shared<AlgorithmNeal8>();
  };
  factory.add_builder("N2", Neal2builder);
  factory.add_builder("N8", Neal8builder);
}

#endif  // BAYESMIX_ALGORITHMS_LOAD_ALGORITHMS_HPP_
