#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  // [0]main [1]data [2]algo [3]coll [4]filecollname
  std::cout << "Running run.cpp" << std::endl;

  std::string type_Mixing = argv[1];

  Factory<BaseMixing> &factory_Mixing = Factory<BaseMixing>::Instance();
  auto mixture = factory_Mixing.create_object(type_Mixing);
  mixture->print_id();

  std::cout << "End of run.cpp" << std::endl;
  return 0;
}
