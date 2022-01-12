#include <matplot/matplot.h>

#include "../lib/argparse/argparse.h"
#include "../src/utils/io_utils.h"

bool check_args(const argparse::ArgumentParser &args) {
  if (args["--n-cl-trace-plot"] != std::string("\"\"")) {
    bayesmix::check_file_is_writeable(
        args.get<std::string>("--n-cl-trace-plot"));
  }
  if (args["--n-cl-hist-plot"] != std::string("\"\"")) {
    bayesmix::check_file_is_writeable(
        args.get<std::string>("--n-cl-hist-plot"));
  }
  return true;
}

int main(int argc, char const *argv[]) {
  argparse::ArgumentParser args("bayesmix::plot");

  args.add_argument("--grid-file")
      .default_value(std::string("\"\""))
      .help(
          "Path to a .csv file containing the grid of points (one per row) "
          "on which the log-density has been evaluated");

  args.add_argument("--dens-file")
      .default_value(std::string("\"\""))
      .help(
          "Path to a .csv file containing the evaluations of the log-density");

  args.add_argument("--n-cl-file")
      .default_value(std::string("\"\""))
      .help(
          "Path to a .csv file containing the number of clusters "
          "(one per row) at each iteration");

  args.add_argument("--dens-plot")
      .default_value(std::string("\"\""))
      .help("File to which to save the density plot");

  args.add_argument("--n-cl-trace-plot")
      .default_value(std::string("\"\""))
      .help(
          "File to which to save the traceplot of the number of clusters "
          "in the MCMC chain");

  args.add_argument("--n-cl-hist-plot")
      .default_value(std::string("\"\""))
      .help(
          "File to which to save the histogram of the number of clusters "
          "in the MCMC chain");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  std::cout << "Running plot_mcmc.cc" << std::endl;
  check_args(args);

  // DENSITY PLOT
  std::string grid_file = args.get<std::string>("--grid-file");
  std::string dens_file = args.get<std::string>("--dens-file");
  std::string dens_plot = args.get<std::string>("--dens-plot");
  if (grid_file != "" and dens_file != "" and dens_plot != "") {
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
      std::cout << "Computing mean density across " << n_iters << " rows..."
                << std::endl;
      dens = dens.array().exp();
      Eigen::MatrixXd mean_dens = dens.colwise().mean();

      // Plot 1D density
      if (dim == 1) {
        std::vector<double> grid_vec(grid.data(), grid.data() + n_points);
        std::vector<double> mean_dens_vec(mean_dens.data(),
                                          mean_dens.data() + n_points);
        matplot::plot(grid_vec, mean_dens_vec);
        std::stringstream title;
        title << "Density estimation on " << n_iters << " iterations";
        matplot::title(title.str());
        matplot::xlabel("Grid");
        matplot::ylabel("Density");
        matplot::save(args.get<std::string>("--dens-plot"));
        std::cout << "Saved density plot to " << dens_plot << std::endl;
      }
      // Plot 2D density
      else {
        // TODO
      }
    }
  }

  // Get other arguments
  std::string ncl_file = args.get<std::string>("--n-cl-file");
  std::string ncl_trace_plot = args.get<std::string>("--n-cl-trace-plot");
  std::string ncl_hist_plot = args.get<std::string>("--n-cl-hist-plot");

  // TRACEPLOT OF NUMBER OF CLUSTERS
  if (ncl_file != "" and ncl_trace_plot != "") {
    bayesmix::check_file_is_writeable(ncl_trace_plot);
    Eigen::MatrixXd num_clus = bayesmix::read_eigen_matrix(ncl_file);
    int n_iters = num_clus.rows();
    std::vector<double> num_clus_vec(num_clus.data(),
                                     num_clus.data() + n_iters);
    std::vector<double> iters_vec(n_iters);
    for (int i = 0; i < n_iters; i++) {
      iters_vec[i] = i;
    }
    matplot::plot(iters_vec, num_clus_vec);
    matplot::title("Traceplot of number of clusters from the MCMC");
    matplot::xlabel("MCMC iterations");
    matplot::ylabel("Number of clusters");
    matplot::save(ncl_trace_plot);
    std::cout << "Saved traceplot to " << ncl_trace_plot << std::endl;
  }

  // HISTOGRAM OF NUMBER OF CLUSTERS
  if (ncl_file != "" and ncl_hist_plot != "") {
    bayesmix::check_file_is_writeable(ncl_hist_plot);
    Eigen::MatrixXd num_clus = bayesmix::read_eigen_matrix(ncl_file);
    int n_iters = num_clus.rows();
    std::vector<double> num_clus_vec(num_clus.data(),
                                     num_clus.data() + n_iters);
    matplot::hist(num_clus_vec);
    matplot::title("Distribution of number of clusters from the MCMC");
    matplot::xlabel("Number of clusters");
    matplot::ylabel("Absolute frequency in the MCMC");
    matplot::save(ncl_hist_plot);
    std::cout << "Saved histogram to " << ncl_hist_plot << std::endl;
  }

  // TODO args.get vs []
  // TODO "" vs std::string("\"\"")
  // TODO check that removing arguments still works

  std::cout << "End of plot_mcmc.cc" << std::endl;
}
