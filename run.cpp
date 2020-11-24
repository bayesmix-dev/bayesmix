#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

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
  std::string datafile = argv[10];
  std::string gridfile = argv[11];
  std::string densfile = argv[12];
  std::string massfile = argv[13];
  std::string nclufile = argv[14];
  std::string clusfile = argv[15];

  // Create factories and objects
  auto &factory_algo = Factory<BaseAlgorithm>::Instance();
  auto &factory_hier = Factory<BaseHierarchy>::Instance();
  auto &factory_mixing = Factory<BaseMixing>::Instance();
  auto algo = factory_algo.create_object(algo_type);
  auto hier = factory_hier.create_object(hier_type);
  auto mixing = factory_mixing.create_object(mix_type);
  auto *coll = new MemoryCollector<bayesmix::MarginalState>();

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

  // Initialize RNG object
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(rng_seed);

  // Read data matrices
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(gridfile);

  // Set algorithm parameters
  algo->set_maxiter(maxiter);
  algo->set_burnin(burnin);

  // Allocate objects in algorithm
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_initial_clusters(hier, init_num_cl);
  if (algo_type == "N8") {
    algo->set_n_aux(3);
  }

  // Run algorithm and density evaluation
  algo->run(coll);
  std::cout << "Computing log-density..." << std::endl;
  Eigen::MatrixXd dens = algo->eval_lpdf(grid, coll);
  std::cout << "Done" << std::endl;
  bayesmix::write_matrix_to_file(dens, densfile);
  std::cout << "Successfully wrote density to " << densfile << std::endl;

  // Collect mixing and cluster states
  auto chain = coll->get_chain();
  Eigen::VectorXd masses(chain.size());
  Eigen::MatrixXd clusterings(chain.size(), data.rows());
  Eigen::VectorXd num_clust(chain.size());
  for (int i = 0; i < chain.size(); i++) {
    for (int j = 0; j < data.rows(); j++) {
      clusterings(i, j) = chain[i].cluster_allocs(j);
    }
    num_clust(i) = chain[i].cluster_states_size();
    bayesmix::MixingState mixstate = chain[i].mixing_state();
    if (mixstate.has_dp_state()) {
      masses(i) = mixstate.dp_state().totalmass();
    }
  }
  // Write collected data to files
  bayesmix::write_matrix_to_file(masses, massfile);
  std::cout << "Successfully wrote total masses to " << massfile << std::endl;
  bayesmix::write_matrix_to_file(num_clust, nclufile);
  std::cout << "Successfully wrote cluster sizes to " << nclufile << std::endl;

  // // Compute cluster estimate
  // std::cout << "Computing cluster estimate..." << std::endl;
  // Eigen::VectorXd clust_est = bayesmix::cluster_estimate(clusterings);
  // std::cout << "Done" << std::endl;
  // bayesmix::write_matrix_to_file(clust_est, clusfile);
  // std::cout << "Successfully wrote clustering to " << clusfile << std::endl;

  std::cout << "End of run.cpp" << std::endl;
  delete coll;
  return 0;
}
