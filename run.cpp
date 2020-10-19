#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  std::string type_mixing = argv[1];
  std::string type_hier  = argv[2];
  std::string type_hypers = argv[3];

  Factory<BaseMixing> &factory_mixing = Factory<BaseMixing>::Instance();
  Factory<HierarchyBase> &factory_hypers = Factory<HierarchyBase>::Instance();
  Factory<HypersBase> &factory_hypers = Factory<HypersBase>::Instance();
  auto mixing = factory_mixing.create_object(type_mixing);
  auto hier   = factory_hier.create_object(type_hier)
  auto hypers = factory_hypers.create_object(type_hypers);
  mixing->print_id();
  hier->print_id();
  hypers->print_id();

  std::cout << "End of run.cpp" << std::endl;
  return 0;
}
