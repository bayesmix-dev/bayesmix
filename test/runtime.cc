// Checks that all the combinations of Algorithms, Mixings and Hierarchies
// can be combined into a sampleable model.

#include <gtest/gtest.h>

#include "src/includes.h"

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
          algo->set_initial_clusters(hier);
        else if (!algo->requires_conjugate_hierarchy())
          algo->set_initial_clusters(hier);
      }
    }
  }
}