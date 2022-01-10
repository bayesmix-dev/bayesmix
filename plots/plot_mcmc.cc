#include <matplot/matplot.h>

#include "../lib/argparse/argparse.h"
#include "../src/utils/io_utils.h"

int main(int argc, char const *argv[]) {
  argparse::ArgumentParser args("bayesmix::plot");

  args.add_argument("--grid-file")
      .required()
      .help(
          "Path to a .csv file containing the grid of points (one per row) "
          "on which the (log) predictive density has been evaluated");

  args.add_argument("--dens-file")
      .required()
      .help(
          "Path to a .csv file containing the evaluations of the (log) "
          "predictive density");

  args.add_argument("--dens-file")
      .default_value(std::string("\"\""))
      .help(
          "(Optional) Where to store the output of the (log) predictive "
          "density");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  std::cout << "Running plot_mcmc.cc" << std::endl;
  std::cout << "End of plot_mcmc.cc" << std::endl;
}
