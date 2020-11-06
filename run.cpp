#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  // Console parameters (temporarily assigned at compile-time)
  //std::string type_mixing = "PY";
  //std::string type_hier = "NNW";
  std::string type_algo = "N2";
  std::string datafile = "resources/data_multi.csv";
  std::string gridfile = "resources/grid_multi.csv";
  std::string densfile = "resources/dens_multi.csv";
  unsigned int init = 0;

  // Create factories and objects
  //Factory<BaseMixing> &factory_mixing = Factory<BaseMixing>::Instance();
  //Factory<HierarchyBase> &factory_hier = Factory<HierarchyBase>::Instance();
  Factory<Algorithm> &factory_algo = Factory<Algorithm>::Instance();
  //auto mixing = factory_mixing.create_object(type_mixing);
  auto mixing = std::make_shared<DirichletMixing>();
  //auto hier = factory_hier.create_object(type_hier);
  auto hier = std::make_shared<HierarchyNNW>();
  auto algo = factory_algo.create_object(type_algo);

  // Set parameters

  // // NNIG
  // hier->set_mu0(5.0);
  // hier->set_lambda(0.1);
  // hier->set_alpha0(2.0);
  // hier->set_beta0(2.0);

  // NNW
  Eigen::Matrix<double, 1, 2> mu0; mu0 << 5.5, 5.5;
  hier->set_mu0(mu0);
  hier->set_lambda(0.2);
  double nu = 5.0;
  hier->set_nu(nu);
  Eigen::MatrixXd tau0 = (1 / nu) * Eigen::Matrix<double, 2, 2>::Identity();
  hier->set_tau0(tau0);
  
  mixing->set_totalmass(1.0);
  algo->set_maxiter(10);
  algo->set_burnin(0);

  // Other objects
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(gridfile);

  // STUFF:
  // Object allocation
  algo->set_mixing(mixing);
  algo->set_data_and_initial_clusters(data, hier, init);
  if(type_algo == "N8") algo->set_n_aux(3);
  BaseCollector *coll = new MemoryCollector();

  // Execution
  algo->print_id();
  algo->get_mixing_id();
  algo->get_hier_id();

  algo->run(coll);
  Eigen::MatrixXd dens = algo->eval_lpdf(grid, coll);
  bayesmix::write_matrix_to_file(dens, densfile);

  std::cout << "End of run.cpp" << std::endl;
  return 0;
}
