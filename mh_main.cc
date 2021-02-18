#include <iostream>

#include "src/algorithms/metropolis.h"
#include "src/utils/io_utils.h"
#include "src/utils/rng.h"

int main(int argc, char *argv[]) {
  std::cout << "Running mh_run.cc" << std::endl;

  std::string datafile = "resources/test/mh_data.csv";
  std::string covsfile = "resources/test/mh_covs.csv";

  Metropolis metro;
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  metro.set_data(data);
  Eigen::MatrixXd covariates = bayesmix::read_eigen_matrix(covsfile);
  metro.set_covariates(covariates);

  auto &rng = bayesmix::Rng::Instance().get();
  rng.seed(20201124);

  std::cout << "With normal MH:" << std::endl;
  metro.run(false);

  // std::cout << std::endl << "With MALA:" << std::endl;
  // metro.run(true);

  std::cout << "End of mh_run.cc" << std::endl;
  return 0;
}
