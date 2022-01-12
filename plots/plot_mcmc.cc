#include <matplot/matplot.h>

#include "../lib/argparse/argparse.h"
#include "../src/utils/io_utils.h"

int main(int argc, char const *argv[]) {
  argparse::ArgumentParser args("bayesmix::plot");

  args.add_argument("--grid-file")
      .required()
      .help(
          "Path to a .csv file containing the grid of points (one per row) "
          "on which the log-density has been evaluated");

  args.add_argument("--dens-file")
      .required()
      .help(
          "Path to a .csv file containing the evaluations of the log-density");

  // args.add_argument("--dens-file")
  //     .default_value(std::string("\"\""))
  //     .help("...");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  std::cout << "Running plot_mcmc.cc" << std::endl;

  // Read relevant matrices
  std::cout << "Reading " << args.get<std::string>("--grid-file") << "..."
            << std::endl;
  Eigen::MatrixXd grid =
      bayesmix::read_eigen_matrix(args.get<std::string>("--grid-file"));
  std::cout << "Reading " << args.get<std::string>("--dens-file") << "..."
            << std::endl;
  Eigen::MatrixXd dens =
      bayesmix::read_eigen_matrix(args.get<std::string>("--dens-file"));

  // Go from log-densities to mean density
  std::cout << "Turning log-density into density..." << std::endl;
  dens = dens.array().exp();
  std::cout << "Computing mean density across " << dens.rows() << " rows..."
            << std::endl;
  Eigen::MatrixXd mean_dens = dens.colwise().mean();

  // Plot density
  std::vector<double> grid_vec(grid.data(), grid.data() + dens.cols());
  std::vector<double> mean_dens_vec(mean_dens.data(),
                                    mean_dens.data() + mean_dens.cols());
  matplot::plot(grid_vec, mean_dens_vec);
  matplot::save("density.png");

  // TODO custom path (including extension) of output plot files
  // TODO title and other goodies
  // TODO check consistent dimensions
  // TODO 1D and 2D density cases
  // TODO traceplot for the MCMC chain of the number of clusters
  // TODO barplot of the distribution of the number of clusters

  std::cout << "End of plot_mcmc.cc" << std::endl;
}
