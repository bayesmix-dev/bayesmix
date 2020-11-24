#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  // Console parameters (temporarily assigned at compile-time)
  std::string algo_type = argv[1];
  // std::string hier_type = "NNW";
  std::string mixing_type = argv[3];
  std::string mixing_args = argv[5];
  std::string datafile = "resources/data_uni.csv";  // TEST
  std::string gridfile = "resources/grid_uni.csv";  // TEST
  std::string densfile = "resources/dens_uni.csv";  // TEST
  std::string massfile = "resources/mass_uni.csv";  // TEST
  std::string nclufile = "resources/nclu_uni.csv";  // TEST
  unsigned int init = 0;
  unsigned int maxiter = 1000;
  unsigned int burnin = 1;
  int rng_seed = 20201103;

  // Initialize prior protos
  bayesmix::NNIGPrior hier_prior;  // TEST
  // bayesmix::DPPrior mix_prior;

  // Create factories and objects
  auto &factory_algo = Factory<BaseAlgorithm>::Instance();
  auto &factory_hier = Factory<BaseHierarchy>::Instance();
  auto &factory_mixing = Factory<BaseMixing>::Instance();
  auto algo = factory_algo.create_object(algo_type);
  // auto hier = factory_hier.create_object(hier_type);
  auto hier = std::make_shared<NNIGHierarchy>();  // TEST
  auto mixing = factory_mixing.create_object(mixing_type);

  // Initialize RNG object
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(rng_seed);

  // Set mixing hyperprior
  std::string mix_prior_str = "bayesmix." + mixing_type + "Prior";
  auto mix_prior_desc = google::protobuf::DescriptorPool::generated_pool()
                            ->FindMessageTypeByName(mix_prior_str);
  assert(mix_prior_desc != NULL);
  google::protobuf::Message *mix_prior =
      google::protobuf::MessageFactory::generated_factory()
          ->GetPrototype(mix_prior_desc)
          ->New();
  bayesmix::read_proto_from_file(mixing_args, mix_prior);
  mixing->set_prior(*mix_prior);

  // NNIG  //TEST
  // NGG hyperprior
  hier_prior.mutable_ngg_prior()->mutable_mean_prior()->set_mean(5.5);
  hier_prior.mutable_ngg_prior()->mutable_mean_prior()->set_var(2.25);
  hier_prior.mutable_ngg_prior()->mutable_var_scaling_prior()->set_shape(0.2);
  hier_prior.mutable_ngg_prior()->mutable_var_scaling_prior()->set_rate(0.6);
  hier_prior.mutable_ngg_prior()->set_shape(1.5);
  hier_prior.mutable_ngg_prior()->mutable_scale_prior()->set_shape(4.0);
  hier_prior.mutable_ngg_prior()->mutable_scale_prior()->set_rate(2.0);
  // // Fixed values hyperprior
  // hier_prior.mutable_fixed_values()->set_mean(5.0);
  // hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
  // hier_prior.mutable_fixed_values()->set_shape(2.0);
  // hier_prior.mutable_fixed_values()->set_scale(2.0);

  // // NNW  //TEST
  // // NGIW hyperprior
  // Eigen::Vector2d mu00;
  // mu00 << 5.5, 5.5;
  // Eigen::Matrix2d ident = Eigen::Matrix2d::Identity();
  // double nu0 = 5.0;
  // bayesmix::to_proto(
  //     mu00,
  //     hier_prior.mutable_ngiw_prior()->mutable_mean_prior()
  //     ->mutable_mean());
  // bayesmix::to_proto(
  //     ident / nu0,
  //     hier_prior.mutable_ngiw_prior()->mutable_mean_prior()
  //     ->mutable_var());
  // hier_prior.mutable_ngiw_prior()->mutable_var_scaling_prior()
  //     ->set_shape(0.2);
  // hier_prior.mutable_ngiw_prior()->mutable_var_scaling_prior()
  //     ->set_rate(0.6);
  // hier_prior.mutable_ngiw_prior()->set_deg_free(nu0);
  // hier_prior.mutable_ngiw_prior()->mutable_scale_prior()->set_deg_free(nu0);
  // bayesmix::to_proto(
  //     ident * nu0,
  //     hier_prior.mutable_ngiw_prior()->mutable_scale_prior()
  //     ->mutable_scale());
  // // Fixed values hyperprior
  // Eigen::Vector2d mu0;
  // mu0 << 5.5, 5.5;
  // double nu0 = 5.0;
  // Eigen::Matrix2d tau0 = Eigen::Matrix2d::Identity() / nu0;
  // bayesmix::to_proto(mu0,
  // hier_prior.mutable_fixed_values()->mutable_mean());
  // hier_prior.mutable_fixed_values()->set_var_scaling(0.2);
  // hier_prior.mutable_fixed_values()->set_deg_free(nu0);
  // bayesmix::to_proto(tau0,
  // hier_prior.mutable_fixed_values()->mutable_scale());

  // Set parameters
  hier->set_prior(hier_prior);
  algo->set_maxiter(maxiter);
  algo->set_burnin(burnin);

  // Read data objects
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(gridfile);

  // STUFF:
  // Object allocation
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_initial_clusters(hier, init);
  if (algo_type == "N8") {
    algo->set_n_aux(3);
  }
  BaseCollector<bayesmix::MarginalState> *coll =
      new MemoryCollector<bayesmix::MarginalState>();

  // Run algorithm and density evaluation
  algo->run(coll);
  std::cout << "Computing log-density..." << std::endl;
  Eigen::MatrixXd dens = algo->eval_lpdf(grid, coll);
  std::cout << "Done" << std::endl;
  bayesmix::write_matrix_to_file(dens, densfile);
  std::cout << "Successfully wrote density to " << densfile << std::endl;

  // Collect mixing states
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
  bayesmix::write_matrix_to_file(masses, massfile);
  std::cout << "Successfully wrote total masses to " << massfile << std::endl;
  bayesmix::write_matrix_to_file(num_clust, nclufile);
  std::cout << "Successfully wrote cluster sizes to " << nclufile << std::endl;

  std::cout << "End of run.cpp" << std::endl;
  delete coll;
  return 0;
}
