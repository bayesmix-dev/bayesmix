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

  args.add_argument("--n-cl-file")
      .required()
      .help(
          "Path to a .csv file containing the number of clusters "
          "(one per row) at each iteration");

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
  std::cout << "Reading " << args.get<std::string>("--n-cl-file") << "..."
            << std::endl;
  Eigen::MatrixXd num_clus =
      bayesmix::read_eigen_matrix(args.get<std::string>("--n-cl-file"));
  int dim = grid.cols();
  int n_points = grid.rows();
  int n_iters = dens.rows();
  // TODO check number of points: grid.rows() == dens.cols()
  // TODO check number of iterations: dens.rows() == num_clus.rows()

  // Go from log-densities to mean density
  std::cout << "Turning log-density into density..." << std::endl;
  dens = dens.array().exp();
  std::cout << "Computing mean density across " << n_iters << " rows..."
            << std::endl;
  Eigen::MatrixXd mean_dens = dens.colwise().mean();

  // Plot density
  std::vector<double> grid_vec(grid.data(), grid.data() + n_points);
  std::vector<double> mean_dens_vec(mean_dens.data(),
                                    mean_dens.data() + n_points);
  matplot::plot(grid_vec, mean_dens_vec);
  matplot::save("density.png");

  // Make traceplot of number of clusters
  std::vector<double> num_clus_vec(num_clus.data(), num_clus.data() + n_iters);
  std::vector<double> iters_vec(n_iters);
  for (int i = 0; i < n_iters; i++) {
    iters_vec[i] = i;
  }
  matplot::plot(iters_vec, num_clus_vec);
  matplot::save("traceplot.png");

  // TODO custom path (including extension) of output plot files
  // TODO title, axis labels, and other goodies
  // TODO 1D and 2D density cases
  // TODO traceplot for the MCMC chain of the number of clusters
  // TODO barplot of the distribution of the number of clusters

  std::cout << "End of plot_mcmc.cc" << std::endl;
}
