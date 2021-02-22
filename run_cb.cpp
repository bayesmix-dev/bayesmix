#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "src/includes.hpp"

using namespace std;
using namespace Eigen;

template <typename M>
M load_csv1(const std::string &path) {
  // Use this if csv is separated with ","
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
  // use this if csv is searated with spaces " "
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

  string filename_mcmc =
      argv[1];  // name of the file containing the mcmc samples
  string filename_pe = argv[2];   // name of the file with the point estimate
  string filename_out = argv[3];  // name of the file to save the output
  int loss_type =
      std::stoi(argv[4]);  // 0: BinderLoss, 1: VI, 2: Normalized VI
  double learning_rate =
      stoi(argv[5]);  // a good choice for "learning_rate" is 1 to 5
  cout << "Learning rate: " << learning_rate << endl;

  Eigen::MatrixXi mcmc, pe_tmp;
  Eigen::VectorXi pe;

  mcmc = load_csv2<MatrixXi>(filename_mcmc);
  pe_tmp = load_csv1<MatrixXi>(filename_pe);

  pe = pe_tmp.row(0);
  cout << "Dimension of the point estimate: " << pe_tmp.cols() << endl;

  cout << "Matrix with dimensions : " << mcmc.rows() << "*" << mcmc.cols()
       << " found." << endl;

  CredibleBall CB =
      CredibleBall(static_cast<LOSS_FUNCTION>(loss_type), mcmc, 0.05, pe);
  CB.calculateRegion(learning_rate);
  double r = CB.getRadius();
  cout << "radius: " << r << "\n";

  cout << "Vertical Upper Bound\n";
  Eigen::VectorXi VUB = CB.VerticalUpperBound();
  cout << "VUB cardinality: " << VUB.size() << " \n";
  for (int i = 0; i < VUB.size(); i++) {
    cout << "Index of the VUB element:" << VUB(i) << " ";
  }
  cout << "\n";
  cout << "Vertical Lower Bound\n";
  Eigen::VectorXi VLB = CB.VerticalLowerBound();
  cout << "VLB cardinality: " << VLB.size() << " \n";
  for (int i = 0; i < VLB.size(); i++) {
    cout << "Index of the VLB element:" << VLB(i) << " ";
  }
  cout << "\n";
  cout << "Horizontal Bound\n";
  Eigen::VectorXi HB = CB.HorizontalBound();
  cout << "HB cardinality: " << HB.size() << " \n";
  for (int i = 0; i < HB.size(); i++) {
    cout << "Index of the HB element:" << HB(i) << " ";
  }
  cout << "\n";

  CB.sumary(HB, VUB, VLB, filename_out);

  return 0;
}

