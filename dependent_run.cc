#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.h"
#include "src/utils/eigen_utils.h"

int main(int argc, char *argv[]) {
  std::cout << "Running dependent_run_mcmc.cc" << std::endl;

  // Get console parameters
  std::string algo_params_file = argv[1];
  std::string hier_type = argv[2];
  std::string hier_args = argv[3];
  std::string mix_type = argv[4];
  std::string mix_args = argv[5];
  std::string collname = argv[6];
  std::string datafile = argv[7];
  std::string gridfile = argv[8];
  std::string densfile = argv[9];
  std::string nclufile = argv[10];
  std::string clusfile = argv[11];
  std::string hier_cov_file;
  std::string hier_grid_cov_file;
  std::string mix_cov_file;
  std::string mix_grid_cov_file;
  if (argc >= 14) {
    hier_cov_file = argv[12];
    hier_grid_cov_file = argv[13];
  }
  if (argc >= 16) {
    mix_cov_file = argv[14];
    mix_grid_cov_file = argv[15];
  }

  // Read algorithm settings proto
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(algo_params_file, &algo_proto);

  // Create factories and objects
  auto &factory_algo = AlgorithmFactory::Instance();
  auto &factory_hier = HierarchyFactory::Instance();
  auto &factory_mixing = MixingFactory::Instance();
  auto algo = factory_algo.create_object(algo_proto.algo_id());
  auto hier = factory_hier.create_object(hier_type);
  auto mixing = factory_mixing.create_object(mix_type);
  BaseCollector *coll;
  if (collname == "") {
    coll = new MemoryCollector();
  } else {
    coll = new FileCollector(collname);
  }

  bayesmix::read_proto_from_file(mix_args, mixing->get_mutable_prior());
  bayesmix::read_proto_from_file(hier_args, hier->get_mutable_prior());

  // Read data matrices
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(gridfile);
  Eigen::MatrixXd hier_cov_grid = Eigen::MatrixXd(data.rows(), 0);
  Eigen::MatrixXd mix_cov_grid = Eigen::MatrixXd(data.rows(), 0);
  if (hier->is_dependent()) {
    hier_cov_grid = bayesmix::read_eigen_matrix(hier_grid_cov_file);
  }
  if (mixing->is_dependent()) {
    mix_cov_grid = bayesmix::read_eigen_matrix(mix_grid_cov_file);
  }

  // Set algorithm parameters
  algo->read_params_from_proto(algo_proto);

  // Allocate objects in algorithm
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_hierarchy(hier);

  // Read and set covariates
  if (hier->is_dependent()) {
    Eigen::MatrixXd hier_cov = bayesmix::read_eigen_matrix(hier_cov_file);
    algo->set_hier_covariates(hier_cov);
  }
  if (mixing->is_dependent()) {
    Eigen::MatrixXd mix_cov = bayesmix::read_eigen_matrix(mix_cov_file);
    algo->set_mix_covariates(mix_cov);
  }

  // Run algorithm and density evaluations
  algo->run(coll);
  std::cout << "Computing log-densities..." << std::endl;
  Eigen::MatrixXd dens(mix_cov_grid.rows(), grid.rows());
  for (int i = 0; i < mix_cov_grid.rows(); i++) {
    Eigen::VectorXd dens_mean_i(grid.rows());
    Eigen::MatrixXd tmp =
        algo->eval_lpdf(coll, grid, hier_cov_grid.row(i), mix_cov_grid.row(i));
    for (int j = 0; j < coll->get_size(); j++) {
      dens_mean_i += tmp.row(j);
    }
    dens.row(i) = dens_mean_i / coll->get_size();
  }
  std::cout << "Done" << std::endl;
  bayesmix::write_matrix_to_file(dens, densfile);
  std::cout << "Successfully wrote densities to " << densfile << std::endl;

  std::cout << "End of dependent_run_mcmc.cc" << std::endl;
  delete coll;
  return 0;
}
