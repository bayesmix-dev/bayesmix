#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  std::string type_mixing = argv[1];
  std::string type_hypers = argv[2];

  Factory<BaseMixing> &factory_mixing = Factory<BaseMixing>::Instance();
  Factory<HypersBase> &factory_hypers = Factory<HypersBase>::Instance();
  auto mixture = factory_mixing.create_object(type_mixing);
  auto hypers  = factory_hypers.create_object(type_hypers);
  mixture->print_id();
  hypers->print_id();

  std::cout << "End of run.cpp" << std::endl;
  return 0;
}
