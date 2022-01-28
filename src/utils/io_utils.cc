#include "io_utils.h"
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

Eigen::MatrixXd bayesmix::read_eigen_matrix(const std::string &filename,
                                            const char delim /* = ','*/) {
  // Initialize objects
  unsigned int cols = 0, rows = 0;
  double buffer[MAXBUFSIZE];
  std::ifstream filestream(filename);
  if (!filestream.is_open()) {
    std::string err = "File " + filename + " does not exist";
    throw std::invalid_argument(err);
  }

  // Loop over file lines
  std::string line, entry;
  while (getline(filestream, line, '\n')) {
    unsigned int temp = 0;
    std::stringstream linestream(line);
    while (getline(linestream, entry, delim)) {
      // Place read values into the buffer array
      std::stringstream entrystream(entry);
      entrystream >> buffer[cols * rows + temp++];
    }
    if (temp == 0) {
      continue;
    }
    if (cols == 0) {
      cols = temp;
    }
    rows++;
  }

  filestream.close();

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
                                    const std::string &filename,
                                    const char delim /*= ','*/) {
  using namespace Eigen;
  std::string del;
  del = delim;
  const IOFormat CSVFormat(StreamPrecision, DontAlignCols, del, "\n");
  std::ofstream file(filename.c_str());
  file << mat.format(CSVFormat);
}
