#include "testing_utils.h"

std::shared_ptr<AbstractHierarchy> get_multivariate_nnw_hierarchy(int dim) {
  Eigen::MatrixXd scale = get_spd_matrix(dim);

  double deg_free = dim + 5;
  double var_scaling = 0.1;
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);

  bayesmix::NNWPrior hier_prior;
  bayesmix::to_proto(mean, hier_prior.mutable_fixed_values()->mutable_mean());
  bayesmix::to_proto(scale,
                     hier_prior.mutable_fixed_values()->mutable_scale());
  hier_prior.mutable_fixed_values()->set_var_scaling(var_scaling);
  hier_prior.mutable_fixed_values()->set_deg_free(deg_free);

  std::shared_ptr<AbstractHierarchy> hier = std::make_shared<NNWHierarchy>();
  hier->get_mutable_prior()->CopyFrom(hier_prior);
  hier->initialize();
  return hier;
}

std::shared_ptr<AbstractHierarchy> get_univariate_nnig_hierarchy() {
  bayesmix::NNIGPrior hier_prior;
  hier_prior.mutable_fixed_values()->set_mean(0.0);
  hier_prior.mutable_fixed_values()->set_var_scaling(0.1);
  hier_prior.mutable_fixed_values()->set_shape(2.0);
  hier_prior.mutable_fixed_values()->set_scale(2.0);

  std::shared_ptr<AbstractHierarchy> hier = std::make_shared<NNIGHierarchy>();
  hier->get_mutable_prior()->CopyFrom(hier_prior);
  hier->initialize();
  return hier;
}

std::shared_ptr<AbstractMixing> get_dirichlet_mixing() {
  bayesmix::DPPrior mix_prior;
  double totalmass = 1.0;
  mix_prior.mutable_fixed_value()->set_totalmass(totalmass);
  auto mixing = std::make_shared<DirichletMixing>();
  mixing->get_mutable_prior()->CopyFrom(mix_prior);
  mixing->set_num_components(5);
  return mixing;
}

std::shared_ptr<BaseAlgorithm> get_algorithm(const std::string& id, int dim) {
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(
      "../resources/benchmarks/default_algo_params.asciipb", &algo_proto);
  std::shared_ptr<BaseAlgorithm> algo =
      AlgorithmFactory::Instance().create_object(id);
  algo->read_params_from_proto(algo_proto);
  algo->set_verbose(false);

  std::shared_ptr<AbstractMixing> mixing = get_dirichlet_mixing();
  std::shared_ptr<AbstractHierarchy> hier;
  if (dim == 1) {
    hier = get_univariate_nnig_hierarchy();
  } else {
    // hier = get_multivariate_nnw_hierarchy(dim);
  }
  hier->initialize();
  algo->set_mixing(mixing);
  algo->set_hierarchy(hier);
  return algo;
}

Eigen::MatrixXd get_spd_matrix(int dim) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim + 1, dim);
  Eigen::MatrixXd out =
      A.transpose() * A + Eigen::MatrixXd::Identity(dim, dim);
  return out;
}
