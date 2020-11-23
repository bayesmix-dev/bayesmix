#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  // Console parameters (temporarily assigned at compile-time)
  // std::string type_mixing = "PY";
  // std::string type_hier = "NNW";
  std::string type_algo = "N2";
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
  bayesmix::DPPrior mix_prior;

  // // Fixed total mass
  // double totalmass = 2.0;
  // mix_prior.mutable_fixed_value()->set_value(totalmass);
  // Gamma-prior total mass
  double alpha_mass = 4.0;
  double beta_mass = 2.0;
  mix_prior.mutable_gamma_prior()->set_shape(alpha_mass);
  mix_prior.mutable_gamma_prior()->set_rate(beta_mass);

  // Create factories and objects
  auto &factory_mixing = Factory<BaseMixing>::Instance();
  auto &factory_hier = Factory<BaseHierarchy>::Instance();
  auto &factory_algo = Factory<BaseAlgorithm>::Instance();
  // auto mixing = factory_mixing.create_object(type_mixing);
  auto mixing = std::make_shared<DirichletMixing>();
  // auto hier = factory_hier.create_object(type_hier);
  auto hier = std::make_shared<NNIGHierarchy>();  // TEST
  auto algo = factory_algo.create_object(type_algo);
  // Initialize RNG object
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(rng_seed);

  // Write parameters

  // NNIG  //TEST
  double mu0 = 5.0;
  double lambda0 = 0.1;
  double alpha0 = 2.0;
  double beta0 = 2.0;
  hier_prior.mutable_fixed_values()->set_mean(mu0);
  hier_prior.mutable_fixed_values()->set_var_scaling(lambda0);
  hier_prior.mutable_fixed_values()->set_shape(alpha0);
  hier_prior.mutable_fixed_values()->set_scale(beta0);

  // // NNW  //TEST
  // Eigen::Vector2d mu0; mu0 << 5.5, 5.5;
  // bayesmix::Vector mu0_proto;
  // bayesmix::to_proto(mu0, &mu0_proto);
  // double lambda0 = 0.2;
  // double nu0 = 5.0;
  // Eigen::Matrix2d tau0 = Eigen::Matrix2d::Identity() / nu0;
  // bayesmix::Matrix tau0_proto;
  // bayesmix::to_proto(tau0, &tau0_proto);
  // *hier_prior.mutable_fixed_values()->mutable_mean() = mu0_proto;
  // hier_prior.mutable_fixed_values()->set_var_scaling(lambda0);
  // hier_prior.mutable_fixed_values()->set_deg_free(nu0);
  // *hier_prior.mutable_fixed_values()->mutable_scale() = tau0_proto;

  // Set parameters
  hier->set_prior(hier_prior);
  mixing->set_prior(mix_prior);
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
  if (type_algo == "N8") {
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
    bayesmix::MixingState mixstate = chain[i].mixing_states(0);
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
