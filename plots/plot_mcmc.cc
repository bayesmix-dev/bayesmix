#include <matplot/matplot.h>

#include "lib/argparse/argparse.h"
#include "plots/plot_utils.h"
#include "src/utils/io_utils.h"

#define EMPTYSTR std::string("\"\"")

int main(int argc, char const *argv[]) {
  argparse::ArgumentParser args("bayesmix::plot");

  args.add_argument("--grid-file")
      .default_value(EMPTYSTR)
      .help(
          "Path to a .csv file containing the grid of points (one per row) "
          "on which the log-density has been evaluated");

  args.add_argument("--dens-file")
      .default_value(EMPTYSTR)
      .help(
          "Path to a .csv file containing the evaluations of the log-density");

  args.add_argument("--n-cl-file")
      .default_value(EMPTYSTR)
      .help(
          "Path to a .csv file containing the number of clusters "
          "(one per row) at each iteration");

  args.add_argument("--dens-plot")
      .default_value(EMPTYSTR)
      .help("File to which to save the density plot");

  args.add_argument("--n-cl-trace-plot")
      .default_value(EMPTYSTR)
      .help(
          "File to which to save the traceplot of the number of clusters "
          "in the MCMC chain");

  args.add_argument("--n-cl-bar-plot")
      .default_value(EMPTYSTR)
      .help(
          "File to which to save the barplot with the empirical distribution "
          "of the number of clusters in the MCMC chain");

  std::cout << "Running plot_mcmc.cc" << std::endl;

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  // Get other arguments
  std::string ncl_file = args.get<std::string>("--n-cl-file");
  std::string ncl_trace_plot = args.get<std::string>("--n-cl-trace-plot");
  std::string ncl_bar_plot = args.get<std::string>("--n-cl-bar-plot");

  // TRACEPLOT OF NUMBER OF CLUSTERS
  if (ncl_file != EMPTYSTR and ncl_trace_plot != EMPTYSTR) {
    bayesmix::check_file_is_writeable(ncl_trace_plot);
    Eigen::MatrixXd num_clus = bayesmix::read_eigen_matrix(ncl_file);
    num_clus_trace(num_clus, ncl_trace_plot);
  }

  // HISTOGRAM OF NUMBER OF CLUSTERS
  if (ncl_file != EMPTYSTR and ncl_bar_plot != EMPTYSTR) {
    bayesmix::check_file_is_writeable(ncl_bar_plot);
    Eigen::MatrixXd num_clus = bayesmix::read_eigen_matrix(ncl_file);
    num_clus_bar(num_clus, ncl_bar_plot);
  }

  // DENSITY PLOT
  std::string grid_file = args.get<std::string>("--grid-file");
  std::string dens_file = args.get<std::string>("--dens-file");
  std::string dens_plot = args.get<std::string>("--dens-plot");

  if (grid_file != EMPTYSTR and dens_file != EMPTYSTR and
      dens_plot != EMPTYSTR) {
    bayesmix::check_file_is_writeable(dens_plot);

    // Read relevant matrices
    Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(grid_file);
    Eigen::MatrixXd dens = bayesmix::read_eigen_matrix(dens_file);
    int dim = grid.cols();
    int n_points = grid.rows();
    int n_iters = dens.rows();

    // Check that matrix dimensions are consistent
    if (n_points != dens.cols()) {
      std::stringstream msg;
      msg << "Matrix dimensions are not consistent: rows of grid = "
          << n_points
          << " should be equal to columns of density = " << dens.cols();
      throw std::invalid_argument(msg.str());
    }
    // Check that grid has appropriate dimensions
    if (dim != 1 and dim != 2) {
      std::cout << "Grid has dimension " << dim << ", its density will not "
                << "be plotted" << std::endl;
    } else {
      // Go from log-densities to mean density
      dens = dens.array().exp();
      Eigen::VectorXd mean_dens = dens.colwise().mean();

      // Plot 1D density
      if (dim == 1) {
        density_plot_1d(grid, mean_dens, dens_plot);
      }
      // Plot 2D density
      else {
        density_plot_2d(grid, mean_dens, dens_plot, true);
      }
    }
  }

  std::cout << "End of plot_mcmc.cc" << std::endl;
}
