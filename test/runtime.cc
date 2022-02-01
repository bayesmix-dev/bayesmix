// Checks that all the combinations of Algorithms, Mixings and Hierarchies
// can be combined into a sampleable model.

#include <gtest/gtest.h>

#include "src/includes.h"
#include "src/utils/testing_utils.h"

TEST(can_build, allmodels) {
  auto &factory_algo = AlgorithmFactory::Instance();
  auto &factory_hier = HierarchyFactory::Instance();
  auto &factory_mixing = MixingFactory::Instance();

  for (auto &algo_id : factory_algo.list_of_known_builders()) {
    auto algo = factory_algo.create_object(algo_id);
    for (auto &mix_id : factory_mixing.list_of_known_builders()) {
      auto mix = factory_mixing.create_object(mix_id);
      algo->set_mixing(mix);
      for (auto &hier_id : factory_hier.list_of_known_builders()) {
        auto hier = factory_hier.create_object(hier_id);
        if (hier->is_conjugate() & algo->requires_conjugate_hierarchy())
          algo->set_hierarchy(hier);
        else if (!algo->requires_conjugate_hierarchy())
          algo->set_hierarchy(hier);
      }
    }
  }
}

TEST(clone, algorithm) {
  std::shared_ptr<BaseAlgorithm> algo = get_algorithm("Neal3", 2);
  std::shared_ptr<BaseAlgorithm> algo_clone = algo->clone();

  algo->get_unique_values()[0]->sample_prior();

  ASSERT_FALSE(
      algo->get_unique_values()[0]->get_state_proto()->DebugString() ==
      algo_clone->get_unique_values()[0]->get_state_proto()->DebugString());
}
