#include "io_utils.h"

#include <Eigen/Dense>
#include <fstream>

Eigen::MatrixXd bayesmix::read_eigen_matrix(const std::string &filename) {
  // Initialize objects
  unsigned int cols = 0, rows = 0;
  double buffer[MAXBUFSIZE];
  std::ifstream istr(filename);
  if (!istr.is_open()) {
    std::string err = "File " + filename + " does not exist";
    throw std::invalid_argument(err);
  }

  // Loop over file lines
  while (!istr.eof()) {
    std::string line;
    getline(istr, line);

    unsigned int temp = 0;
    std::stringstream stream(line);
    while (!stream.eof()) {
      // Place read values into the buffer array
      stream >> buffer[cols * rows + temp++];
    }
    if (temp == 0) {
      continue;
    }
    if (cols == 0) {
      cols = temp;
    }
    rows++;
  }

  istr.close();
  rows--;

  // Fill an Eigen Matrix with values from the buffer array
  Eigen::MatrixXd mat(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat(i, j) = buffer[cols * i + j];
    }
  }
  return mat;
};

void bayesmix::write_matrix_to_file(const Eigen::MatrixXd &mat,
                                    const std::string &filename) {
  using namespace Eigen;
  const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
  std::ofstream file(filename.c_str());
  file << mat.format(CSVFormat);
}
