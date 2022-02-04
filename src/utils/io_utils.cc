#include "io_utils.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

bool bayesmix::check_file_is_writeable(const std::string &filename) {
  std::ofstream ofstr;
  ofstr.open(filename);
  if (ofstr.fail()) {
    ofstr.close();
    throw std::invalid_argument("Cannot write to " + filename);
  }
  ofstr.close();
  return true;
}

Eigen::MatrixXd bayesmix::read_eigen_matrix(const std::string &filename,
                                            const char delim /* = ','*/) {
  // Initialize objects
  int rows = 0, cols = 0;
  std::ifstream filestream(filename);
  if (!filestream.is_open()) {
    std::string err = "File " + filename + " does not exist";
    throw std::invalid_argument(err);
  }

  // Get number of rows and columns
  std::string line, entry;
  while (getline(filestream, line, '\n')) {
    rows++;
    if (rows == 1) {
      std::stringstream linestream(line);
      while (getline(linestream, entry, delim)) {
        cols++;
      }
    }
  }

  // Reset file stream to the beginning of the file
  filestream.clear();
  filestream.seekg(0, std::ios::beg);

  // Fill an Eigen Matrix with values from the matrix
  Eigen::MatrixXd mat(rows, cols);
  int i = 0;
  while (getline(filestream, line, '\n')) {
    int j = 0;
    std::stringstream linestream(line);
    while (getline(linestream, entry, delim)) {
      std::stringstream entrystream(entry);
      mat(i, j) = std::stof(entry);
      j++;
    }
    i++;
  }

  filestream.close();
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
