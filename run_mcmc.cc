#include <math.h>

#include <fstream>
#include <iostream>

#include "lib/argparse/argparse.h"
#include "src/includes.h"

bool check_file_is_writeable(std::string filename) {
  std::ofstream ofstr;
  ofstr.open(filename);
  if (ofstr.fail()) {
    std::cerr << "Error: cannot write to " << filename << std::endl;
    ofstr.close();
    return false;
  }
  ofstr.close();
  return true;
}

bool check_args(argparse::ArgumentParser args) {
  if (args["--collname"] != std::string("memory")) {
    check_file_is_writeable(args.get<std::string>("--collname"));
  }
  if (args["--gridfile"] != std::string("\"\"")) {
    check_file_is_writeable(args.get<std::string>("--densfile"));
  }
  if (args["--nclufile"] != std::string("\"\"")) {
    check_file_is_writeable(args.get<std::string>("--nclufile"));
  }
  if (args["--clusfile"] != std::string("\"\"")) {
    check_file_is_writeable(args.get<std::string>("--clusfile"));
  }

  return true;
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser args("bayesmix::run");

  args.add_argument("--algo_params_file")
      .required()
      .help(
          "asciipb file with the parameters of the algorithm, see "
          "the file proto/algorithm_params.proto");

  args.add_argument("--hier_type")
      .required()
      .help(
          "enum string of the hierarchy, see the file "
          "proto/hierarchy_id.proto");

  args.add_argument("--hier_args")
      .required()
      .help(
          "asciipb file with the parameters of the hierarchy, see "
          "the file proto/hierarchy_prior.proto");

  args.add_argument("--mix_type")
      .required()
      .help("enum string of the mixing, see the file proto/mixing_id.proto");

  args.add_argument("--mix_args")
      .required()
      .help(
          "asciipb file with the parameters of the mixing, see "
          "the file proto/mixing_prior.proto");

  args.add_argument("--collname")
      .required()
      .default_value("memory")
      .help("If not 'memory', the path where to save the MCMC chains");

  args.add_argument("--datafile")
      .required()
      .help("Path to a .csv file containing the observations (one per row)");

  args.add_argument("--gridfile")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Path to a csv file containin a grid of points where to "
          "evaluate the (log) predictive density");

  args.add_argument("--densfile")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the output of the (log) predictive "
          "density");

  args.add_argument("--nclufile")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the MCMC chain of the number of "
          "clusters");

  args.add_argument("--clusfile")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the MCMC chain of the cluster "
          "allocations");

  args.add_argument("--bestclusfile")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the best cluster allocation found by "
          "minimizing the Binder loss funciton over the visited partitions");

  args.add_argument("--hier_cov_file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models. Path to a csv file with the "
          "covariates used in the hierarchy");

  args.add_argument("--hier_grid_cov_file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models and when 'gridfile' is not "
          "empty. "
          "Path to a csv file with the values covariates used in the "
          "hierarchy "
          "on which to evaluate the (log) predictive density");

  args.add_argument("--mix_cov_file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models. Path to a csv file with the "
          "covariates used in the mixing");

  args.add_argument("--mix_grid_cov_file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Only for dependent models and when 'gridfile' is not "
          "empty. "
          "Path to a csv file with the values covariates used in the mixing "
          "on which to evaluate the (log) predictive density");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  std::cout << "Running run_mcmc.cc" << std::endl;
  check_args(args);

  // Read algorithm settings proto
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(args.get<std::string>("--algo_params_file"),
                                 &algo_proto);

  // Create factories and objects
  auto &factory_algo = AlgorithmFactory::Instance();
  auto &factory_hier = HierarchyFactory::Instance();
  auto &factory_mixing = MixingFactory::Instance();
  auto algo = factory_algo.create_object(algo_proto.algo_id());
  auto hier = factory_hier.create_object(args.get<std::string>("--hier_type"));
  auto mixing =
      factory_mixing.create_object(args.get<std::string>("--mix_type"));

  BaseCollector *coll;
  if (args["--collname"] == std::string("memory")) {
    std::cout << "Creating MemoryCollector" << std::endl;
    coll = new MemoryCollector();
  } else {
    std::cout << "Creating FileCollector, writing to file: "
              << args.get<std::string>("--collname") << std::endl;
    coll = new FileCollector(args.get<std::string>("--collname"));
  }

  bayesmix::read_proto_from_file(args.get<std::string>("--mix_args"),
                                 mixing->get_mutable_prior());
  bayesmix::read_proto_from_file(args.get<std::string>("--hier_args"),
                                 hier->get_mutable_prior());

  // Read data matrices
  Eigen::MatrixXd data =
      bayesmix::read_eigen_matrix(args.get<std::string>("--datafile"));

  // Set algorithm parameters
  algo->read_params_from_proto(algo_proto);

  // Allocate objects in algorithm
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_hierarchy(hier);

  // Read and set covariates
  if (hier->is_dependent()) {
    Eigen::MatrixXd hier_cov =
        bayesmix::read_eigen_matrix(args.get<std::string>("--hier_cov_file"));
    algo->set_hier_covariates(hier_cov);
  }

  if (mixing->is_dependent()) {
    Eigen::MatrixXd mix_cov =
        bayesmix::read_eigen_matrix(args.get<std::string>("--mix_cov_file"));
    algo->set_mix_covariates(mix_cov);
  }

  // Run algorithm
  algo->run(coll);

  if (args["--gridfile"] != std::string("\"\"")) {
    Eigen::MatrixXd grid =
        bayesmix::read_eigen_matrix(args.get<std::string>("--gridfile"));
    Eigen::MatrixXd hier_cov_grid = Eigen::RowVectorXd(0);
    Eigen::MatrixXd mix_cov_grid = Eigen::RowVectorXd(0);
    if (hier->is_dependent()) {
      hier_cov_grid = bayesmix::read_eigen_matrix(
          args.get<std::string>("--hier_grid_cov_file"));
    }
    if (mixing->is_dependent()) {
      mix_cov_grid = bayesmix::read_eigen_matrix(
          args.get<std::string>("--mix_grid_cov_file"));
    }

    std::cout << "Computing log-density..." << std::endl;
    Eigen::MatrixXd dens =
        algo->eval_lpdf(coll, grid, hier_cov_grid, mix_cov_grid);
    bayesmix::write_matrix_to_file(dens, args.get<std::string>("--densfile"));
    std::cout << "Successfully wrote density to "
              << args.get<std::string>("--densfile") << std::endl;
  }

  if ((args.get<std::string>("--nclufile") != std::string("\"\"")) ||
      (args.get<std::string>("--clusfile") != std::string("\"\"")) ||
      (args.get<std::string>("--bestclusfile") != std::string("\"\""))) {
    Eigen::MatrixXi clusterings(coll->get_size(), data.rows());
    Eigen::VectorXi num_clust(coll->get_size());
    for (int i = 0; i < coll->get_size(); i++) {
      bayesmix::AlgorithmState state;
      coll->get_next_state(&state);
      for (int j = 0; j < data.rows(); j++) {
        clusterings(i, j) = state.cluster_allocs(j);
      }
      num_clust(i) = state.cluster_states_size();
    }

    if (args.get<std::string>("--nclufile") != std::string("\"\"")) {
      bayesmix::write_matrix_to_file(num_clust,
                                     args.get<std::string>("--nclufile"));
      std::cout << "Successfully wrote number of clusters to "
                << args.get<std::string>("--nclufile") << std::endl;
    }

    if (args.get<std::string>("--clusfile") != std::string("\"\"")) {
      bayesmix::write_matrix_to_file(clusterings,
                                     args.get<std::string>("--clusfile"));
      std::cout << "Successfully wrote cluster allocations to "
                << args.get<std::string>("--clusfile") << std::endl;
    }

    if (args.get<std::string>("--bestclusfile") != std::string("\"\"")) {
      Eigen::VectorXi best_clus = bayesmix::cluster_estimate(clusterings);
      bayesmix::write_matrix_to_file(best_clus,
                                     args.get<std::string>("--bestclusfile"));
      std::cout << "Successfully wrote best cluster allocations to "
                << args.get<std::string>("--bestclusfile") << std::endl;
    }
  }

  std::cout << "End of run_mcmc.cc" << std::endl;
  delete coll;
  return 0;
}
