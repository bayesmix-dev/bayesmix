#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.h"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cc" << std::endl;

  // Get console parameters
  std::string algo_type = argv[1];
  int rng_seed = std::stoi(argv[2]);
  unsigned int init_num_cl = std::stoi(argv[3]);
  unsigned int maxiter = std::stoi(argv[4]);
  unsigned int burnin = std::stoi(argv[5]);
  std::string hier_type = argv[6];
  std::string hier_args = argv[7];
  std::string mix_type = argv[8];
  std::string mix_args = argv[9];
  std::string collname = argv[10];
  std::string datafile = argv[11];
  std::string gridfile = argv[12];
  std::string densfile = argv[13];
  std::string massfile = argv[14];
  std::string nclufile = argv[15];
  std::string clusfile = argv[16];
  std::string hier_cov_file;
  std::string grid_cov_file;
  if (argc >= 18) {
    hier_cov_file = argv[17];
  }
  if (argc >= 19) {
    grid_cov_file = argv[18];
  }

  // Create factories and objects
  auto &factory_algo = Factory<BaseAlgorithm>::Instance();
  auto &factory_hier = Factory<BaseHierarchy>::Instance();
  auto &factory_mixing = Factory<BaseMixing>::Instance();
  auto algo = factory_algo.create_object(algo_type);
  auto hier = factory_hier.create_object(hier_type);
  auto mixing = factory_mixing.create_object(mix_type);
  BaseCollector *coll;
  if (collname == "") {
    coll = new MemoryCollector();
  } else {
    coll = new FileCollector(collname);
  }

  // Set mixing hyperprior
  std::string mix_prior_str = "bayesmix." + mix_type + "Prior";
  auto mix_prior_desc = google::protobuf::DescriptorPool::generated_pool()
                            ->FindMessageTypeByName(mix_prior_str);
  assert(mix_prior_desc != NULL);
  auto *mix_prior = google::protobuf::MessageFactory::generated_factory()
                        ->GetPrototype(mix_prior_desc)
                        ->New();
  bayesmix::read_proto_from_file(mix_args, mix_prior);
  mixing->set_prior(*mix_prior);

  // Set hierarchies hyperprior
  std::string hier_prior_str = "bayesmix." + hier_type + "Prior";
  auto hier_prior_desc = google::protobuf::DescriptorPool::generated_pool()
                             ->FindMessageTypeByName(hier_prior_str);
  assert(hier_prior_desc != NULL);
  auto *hier_prior = google::protobuf::MessageFactory::generated_factory()
                         ->GetPrototype(hier_prior_desc)
                         ->New();
  bayesmix::read_proto_from_file(hier_args, hier_prior);
  hier->set_prior(*hier_prior);
  hier->initialize();

  // Initialize RNG object
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(rng_seed);

  // Read data matrices
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(gridfile);
  Eigen::MatrixXd cov_grid(0, 0);
  if (hier->is_dependent()) {
    cov_grid = bayesmix::read_eigen_matrix(grid_cov_file);
  }

  // Set algorithm parameters
  algo->set_maxiter(maxiter);
  algo->set_burnin(burnin);

  // Allocate objects in algorithm
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_initial_clusters(hier, init_num_cl);
  if (algo_type == "Neal8") {
    auto algocast = std::dynamic_pointer_cast<Neal8Algorithm>(algo);
    algocast->set_n_aux(3);
  }

  // Read and set covariates
  if (hier->is_dependent()) {
    Eigen::MatrixXd hier_cov = bayesmix::read_eigen_matrix(hier_cov_file);
    algo->set_hier_covariates(hier_cov);
  }

  // Run algorithm and density evaluation
  algo->run(coll);
  std::cout << "Computing log-density..." << std::endl;
  Eigen::MatrixXd dens;
  if (hier->is_dependent()) {
    dens = algo->eval_lpdf(grid, cov_grid, coll);
  } else {
    dens = algo->eval_lpdf(grid, coll);
  }
  std::cout << "Done" << std::endl;
  bayesmix::write_matrix_to_file(dens, densfile);
  std::cout << "Successfully wrote density to " << densfile << std::endl;

  // Collect mixing and cluster states
  Eigen::VectorXd masses(coll->get_size());
  Eigen::MatrixXd clusterings(coll->get_size(), data.rows());
  Eigen::VectorXd num_clust(coll->get_size());
  for (int i = 0; i < coll->get_size(); i++) {
    bayesmix::MarginalState state;
    coll->get_next_state(&state);
    for (int j = 0; j < data.rows(); j++) {
      clusterings(i, j) = state.cluster_allocs(j);
    }
    num_clust(i) = state.cluster_states_size();
    bayesmix::MixingState mixstate = state.mixing_state();
    if (mixstate.has_dp_state()) {
      masses(i) = mixstate.dp_state().totalmass();
    }
  }
  // Write collected data to files
  bayesmix::write_matrix_to_file(masses, massfile);
  std::cout << "Successfully wrote total masses to " << massfile << std::endl;
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
