#ifndef IO_UTILS_HPP
#define IO_UTILS_HPP

#include <Eigen/Dense>
#include <fstream>

#define MAXBUFSIZE ((int)1e6)

namespace bayesmix {
//! Returns an Eigen Matrix after reading it from a file.
Eigen::MatrixXd read_eigen_matrix(const std::string &filename);
void write_matrix_to_file(const Eigen::MatrixXd &mat,
                          std::string filename = "resources/mat.csv");
}  // namespace bayesmix

#endif  // IO_UTILS_HPP
