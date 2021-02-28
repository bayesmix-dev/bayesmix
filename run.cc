#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.h"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cc" << std::endl;

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
  Eigen::MatrixXd hier_cov_grid = Eigen::RowVectorXd(0);
  Eigen::MatrixXd mix_cov_grid = Eigen::RowVectorXd(0);
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

  // Run algorithm and density evaluation
  algo->run(coll);
  std::cout << "Computing log-density..." << std::endl;
  Eigen::MatrixXd dens =
      algo->eval_lpdf(coll, grid, hier_cov_grid, mix_cov_grid);
  std::cout << "Done" << std::endl;
  bayesmix::write_matrix_to_file(dens, densfile);
  std::cout << "Successfully wrote density to " << densfile << std::endl;

  // Collect mixing and cluster states
  Eigen::VectorXd masses(coll->get_size());
  Eigen::MatrixXd clusterings(coll->get_size(), data.rows());
  Eigen::VectorXd num_clust(coll->get_size());
  for (int i = 0; i < coll->get_size(); i++) {
    bayesmix::AlgorithmState state;
    coll->get_next_state(&state);
    for (int j = 0; j < data.rows(); j++) {
      clusterings(i, j) = state.cluster_allocs(j);
    }
    num_clust(i) = state.cluster_states_size();
  }
  // Write collected data to files
  bayesmix::write_matrix_to_file(num_clust, nclufile);
  std::cout << "Successfully wrote cluster sizes to " << nclufile << std::endl;
  // Compute cluster estimate
  std::cout << "Computing cluster estimate..." << std::endl;
  Eigen::VectorXd clust_est = bayesmix::cluster_estimate(clusterings);
  std::cout << "Done" << std::endl;
  bayesmix::write_matrix_to_file(clust_est, clusfile);
  std::cout << "Successfully wrote clustering to " << clusfile << std::endl;

  std::cout << "End of run.cc" << std::endl;
  delete coll;
  return 0;
}
