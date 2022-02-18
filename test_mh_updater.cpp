#include <math.h>

#include <fstream>
#include <iostream>

#include "lib/argparse/argparse.h"
#include "src/includes.h"

int main() {
  // Define prior hypers
  bayesmix::AlgorithmState::HierarchyHypers hypers_proto;
  hypers_proto.mutable_nnig_state()->set_mean(0.0);
  hypers_proto.mutable_nnig_state()->set_var_scaling(0.1);
  hypers_proto.mutable_nnig_state()->set_shape(4.0);
  hypers_proto.mutable_nnig_state()->set_scale(3.0);

  bayesmix::NNIGPrior hier_prior;
  hier_prior.mutable_fixed_values()->set_mean(0.0);
  hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
  hier_prior.mutable_fixed_values()->set_shape(4.0);
  hier_prior.mutable_fixed_values()->set_scale(3.0);

  auto prior = std::make_shared<NIGPriorModel>();
  prior->get_mutable_prior()->CopyFrom(hier_prior);

  // prior->set_hypers_from_proto(hypers_proto);
  auto like = std::make_shared<UniNormLikelihood>();
  auto updater = std::make_shared<MalaUpdater>(0.001);
  auto hier = std::make_shared<NNIGHierarchy>();
  hier->set_likelihood(like);
  hier->set_prior(prior);
  hier->set_updater(updater);
  std::cout << "here" << std::endl;

  hier->initialize();
  std::cout << "initializing" << std::endl;

  auto& rng = bayesmix::Rng::Instance().get();
  int ndata = 250;
  Eigen::VectorXd data(ndata);
  for (int i = 0; i < ndata; i++) {
    data(i) = stan::math::normal_rng(5, 1.0, rng);
    hier->add_datum(i, data.row(i));
  }

  int niter = 10000;
  Eigen::MatrixXd chain(niter, 2);
  for (int i = 0; i < niter; i++) {
    hier->sample_full_cond();
    chain.row(i) = hier->get_state().get_unconstrained();
  }

  bayesmix::write_matrix_to_file(chain, "mcmc_chain_test.csv");
}
