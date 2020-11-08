#ifndef LOAD_ALGORITHMS_HPP
#define LOAD_ALGORITHMS_HPP

#include "../runtime/Factory.hpp"

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

#endif  // LOAD_ALGORITHMS_HPP
