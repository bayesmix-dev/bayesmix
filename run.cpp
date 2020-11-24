#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  // Algorithm parameters
  unsigned int init = 0;
  unsigned int maxiter = 1000;
  unsigned int burnin = 1;
  int rng_seed = 20201103;

  // Get console parameters
  std::string algo_type = argv[1];
  std::string hier_type = argv[2];
  std::string hier_args = argv[3];
  std::string mix_type = argv[4];
  std::string mix_args = argv[5];
  std::string datafile = "resources/data_multi.csv";
  std::string gridfile = "resources/grid_multi.csv";
  std::string densfile = "resources/dens_multi.csv";
  std::string massfile = "resources/mass_multi.csv";
  std::string nclufile = "resources/nclu_multi.csv";

  // Create factories and objects
  auto &factory_algo = Factory<BaseAlgorithm>::Instance();
  auto &factory_hier = Factory<BaseHierarchy>::Instance();
  auto &factory_mixing = Factory<BaseMixing>::Instance();
  auto algo = factory_algo.create_object(algo_type);
  auto hier = factory_hier.create_object(hier_type);
  auto mixing = factory_mixing.create_object(mix_type);
  BaseCollector<bayesmix::MarginalState> *coll =
      new MemoryCollector<bayesmix::MarginalState>();

  // Set mixing hyperprior
  std::string mix_prior_str = "bayesmix." + mix_type + "Prior";
  auto mix_prior_desc = google::protobuf::DescriptorPool::generated_pool()
                            ->FindMessageTypeByName(mix_prior_str);
  assert(mix_prior_desc != NULL);
  google::protobuf::Message *mix_prior =
      google::protobuf::MessageFactory::generated_factory()
          ->GetPrototype(mix_prior_desc)
          ->New();
  bayesmix::read_proto_from_file(mix_args, mix_prior);
  mixing->set_prior(*mix_prior);

  // Set hierarchies hyperprior
  std::string hier_prior_str = "bayesmix." + hier_type + "Prior";
  auto hier_prior_desc = google::protobuf::DescriptorPool::generated_pool()
                             ->FindMessageTypeByName(hier_prior_str);
  assert(hier_prior_desc != NULL);
  google::protobuf::Message *hier_prior =
      google::protobuf::MessageFactory::generated_factory()
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
  algo->set_initial_clusters(hier, init);
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
  Eigen::VectorXd num_clust(chain.size());
  for (int i = 0; i < chain.size(); i++) {
    bayesmix::MixingState mixstate = chain[i].mixing_state();
    if (mixstate.has_dp_state()) {
      masses[i] = mixstate.dp_state().totalmass();
      num_clust[i] = chain[i].cluster_states_size();
    }
  }

  // Write mixing and cluster states
  bayesmix::write_matrix_to_file(masses, massfile);
  std::cout << "Successfully wrote total masses to " << massfile << std::endl;
  bayesmix::write_matrix_to_file(num_clust, nclufile);
  std::cout << "Successfully wrote cluster sizes to " << nclufile << std::endl;

  std::cout << "End of run.cpp" << std::endl;
  delete coll;
  return 0;
}
