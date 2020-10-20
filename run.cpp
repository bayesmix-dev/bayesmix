#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  std::string type_mixing = argv[1];
  std::string type_hier   = argv[2];

  Factory<BaseMixing> &factory_mixing = Factory<BaseMixing>::Instance();
  Factory<HierarchyBase> &factory_hier = Factory<HierarchyBase>::Instance();
  auto mixing = factory_mixing.create_object(type_mixing);
  auto hier = factory_hier.create_object(type_hier);
  mixing->print_id();
  hier->print_id();

  std::cout << "End of run.cpp" << std::endl;
  return 0;
}
