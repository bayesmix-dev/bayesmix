#include "base_algorithm.h"

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_params.pb.h"
#include "algorithm_state.pb.h"
#include "lib/progressbar/progressbar.h"
#include "mixing_state.pb.h"
#include "src/algorithms/neal8_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/rng.h"

Eigen::MatrixXd BaseAlgorithm::eval_lpdf(
    BaseCollector *const collector, const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &hier_covariate /*= Eigen::RowVectorXd(0)*/,
    const Eigen::RowVectorXd &mix_covariate /*= Eigen::RowVectorXd(0)*/) {
  std::deque<Eigen::VectorXd> lpdf;
  bool keep = true;
  progresscpp::ProgressBar *bar = nullptr;
  if (verbose) {
    bar = new progresscpp::ProgressBar(collector->get_size(), 60);
  }
  while (keep) {
    keep = update_state_from_collector(collector);
    if (!keep) {
      break;
    }
    lpdf.push_back(lpdf_from_state(grid, hier_covariate, mix_covariate));
    if (verbose) {
      ++(*bar);
      bar->display();
    }
  }
  collector->reset();
  if (verbose) {
    bar->done();
    delete bar;
    print_ending_message();
  }
  return bayesmix::stack_vectors(lpdf);
}

void BaseAlgorithm::read_params_from_proto(
    const bayesmix::AlgorithmParams &params) {
  // Generic parameters
  maxiter = params.iterations();
  burnin = params.burnin();
  init_num_clusters = params.init_num_clusters();
  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(params.rng_seed());
}

void BaseAlgorithm::initialize() {
  if (verbose) {
    std::cout << "Initializing... " << std::flush;
  }
  // Perform checks
  if (data.rows() == 0) {
    throw std::invalid_argument("Data was not provided to algorithm");
  }
  // Hierarchy checks
  if (unique_values.size() == 0) {
    throw std::invalid_argument("Hierarchy was not provided to algorithm");
  }
  if (unique_values[0]->is_conjugate() == false and
      requires_conjugate_hierarchy()) {
    std::string msg = "Algorithm \"" + bayesmix::AlgorithmId_Name(get_id()) +
                      "\"  only supports conjugate hierarchies";
    throw std::invalid_argument(msg);
  }
  if (unique_values[0]->is_multivariate() == false && data.cols() > 1) {
    throw std::invalid_argument(
        "Multivariate data supplied to univariate hierarchy");
  }
  if (hier_covariates.rows() != 0) {
    if (unique_values[0]->is_dependent() == false) {
      throw std::invalid_argument(
          "Covariates supplied to non-dependent hierarchy");
    }
    if (data.rows() != hier_covariates.rows()) {
      throw std::invalid_argument(
          "Sizes of data and hierarchy covariates do not match");
    }
  } else {
    // Create empty covariates vector
    hier_covariates = Eigen::MatrixXd::Zero(data.rows(), 0);
  }
  // Mixing checks
  if (mixing == nullptr) {
    throw std::invalid_argument("Mixing was not provided to algorithm");
  }
  if (this->is_conditional() != mixing->is_conditional()) {
    throw std::invalid_argument(
        "Algorithm and mixing must be either both "
        "marginal or both conditional");
  }
  if (mix_covariates.rows() != 0) {
    if (mixing->is_dependent() == false) {
      throw std::invalid_argument(
          "Covariates supplied to non-dependent mixing");
    }
    if (data.rows() != mix_covariates.rows()) {
      throw std::invalid_argument(
          "Sizes of data and mixing covariates do not match");
    }
  } else {
    // Create empty covariates vector
    mix_covariates = Eigen::MatrixXd::Zero(data.rows(), 0);
  }
  mixing->set_covariates(&mix_covariates);
  // Initialize mixing
  mixing->set_num_components(init_num_clusters);
  mixing->initialize();
  // Interpet default number of clusters
  if (mixing->get_num_components() == 0) {
    mixing->set_num_components(data.rows());
  }
  // Initialize hierarchies
  unique_values[0]->initialize();
  unsigned int num_components = mixing->get_num_components();
  for (size_t i = 0; i < num_components - 1; i++) {
    unique_values.push_back(unique_values[0]->clone());
    unique_values[i]->sample_prior();
  }
  // Build uniform probability on clusters, given their initial number
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distro(0, num_components - 1);
  // Allocate one datum per cluster first, and update cardinalities
  allocations.clear();
  for (size_t i = 0; i < num_components; i++) {
    allocations.push_back(i);
    unique_values[i]->add_datum(i, data.row(i), update_hierarchy_params(),
                                hier_covariates.row(i));
  }
  // Randomly allocate all remaining data, and update cardinalities
  for (size_t i = num_components; i < data.rows(); i++) {
    unsigned int clust = distro(generator);
    allocations.push_back(clust);
    unique_values[clust]->add_datum(i, data.row(i), update_hierarchy_params(),
                                    hier_covariates.row(i));
  }
  if (verbose) {
    std::cout << "Done" << std::endl;
  }
}

void BaseAlgorithm::update_hierarchy_hypers() {
  bayesmix::AlgorithmState::ClusterState clust;
  std::vector<bayesmix::AlgorithmState::ClusterState> states;
  // Build vector of states associated to non-empty clusters
  for (auto &un : unique_values) {
    if (un->get_card() > 0) {
      un->write_state_to_proto(&clust);
      states.push_back(clust);
    }
  }
  unique_values[0]->update_hypers(states);
}

bayesmix::AlgorithmState BaseAlgorithm::get_state_as_proto(unsigned int iter) {
  bayesmix::AlgorithmState iter_out;
  // Transcribe iteration number, allocations, and cardinalities
  iter_out.set_iteration_num(iter);
  *iter_out.mutable_cluster_allocs() = {allocations.begin(),
                                        allocations.end()};
  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    bayesmix::AlgorithmState::ClusterState clusval;
    unique_values[i]->write_state_to_proto(&clusval);
    iter_out.add_cluster_states()->CopyFrom(clusval);
  }
  // Transcribe mixing state
  bayesmix::MixingState mixstate;
  mixing->write_state_to_proto(&mixstate);
  iter_out.mutable_mixing_state()->CopyFrom(mixstate);

  return iter_out;
}

bool BaseAlgorithm::update_state_from_collector(BaseCollector *coll) {
  bool success = coll->get_next_state(&curr_state);
  return success;
}
