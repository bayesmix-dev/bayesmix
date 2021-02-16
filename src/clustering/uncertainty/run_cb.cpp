#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../lib/math/lib/eigen_3.3.9/Eigen/Dense"
#include "../../utils/cluster_utils.hpp"
#include "../../utils/io_utils.hpp"
#include "CredibleBall.hpp"

using namespace std;
using namespace Eigen;

template <typename M>
M load_csv1(const std::string &path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<int> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, RowMajor>>(
      values.data(), rows, values.size() / rows);
}
template <typename M>
M load_csv2(const std::string &path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<int> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ' ')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, RowMajor>>(
      values.data(), rows, values.size() / rows);
}

int main(int argc, char const *argv[]) {
  cout << "Credible-balls test" << endl;

  if (argc != 6) {
    throw domain_error(
        "Syntax : ./run_cb filename_mcmc filename_pe filename_out loss rate");
  }

  string filename_mcmc = argv[1];
  string filename_pe = argv[2];
  string filename_out = argv[3];
  int loss_type = std::stoi(argv[4]);
  double learning_rate = stoi(argv[5]);
  cout << "Learning rate: " << learning_rate << endl;

  Eigen::MatrixXi mcmc, pe_tmp;
  Eigen::VectorXi pe;
  cout << "ok1" << endl;
  pe_tmp = load_csv1<MatrixXi>(filename_pe);
  cout << "ok2" << endl;
  pe = pe_tmp.row(0);
  cout << pe_tmp.cols() << endl;
  cout << "ok3" << endl;
  mcmc = load_csv2<MatrixXi>(filename_mcmc);

  cout << "Matrix with dimensions : " << mcmc.rows() << "*" << mcmc.cols()
       << " found." << endl;

  CredibleBall CB =
      CredibleBall(static_cast<LOSS_FUNCTION>(loss_type), mcmc, 0.05, pe);
  cout << "ok4" << endl;
  double r = CB.calculateRegion(learning_rate);
  cout << "radius: " << r << "\n";

  Eigen::VectorXi VUB = CB.VerticalUpperBound();
  cout << "VUB cardinality: " << VUB.size() << " \n";
  for (int i = 0; i < VUB.size(); i++) {
    cout << "Index of the VUB element:" << VUB(i) << " ";
  }
  cout << "\n";

  Eigen::VectorXi VLB = CB.VerticalLowerBound();
  cout << "VLB cardinality: " << VLB.size() << " \n";
  for (int i = 0; i < VLB.size(); i++) {
    cout << "Index of the VLB element:" << VLB(i) << " ";
  }
  cout << "\n";

  Eigen::VectorXi HB = CB.HorizontalBound();
  cout << "HB cardinality: " << HB.size() << " \n";
  for (int i = 0; i < HB.size(); i++) {
    cout << "Index of the HB element:" << HB(i) << " ";
  }
  cout << "\n";

  return 0;
}
