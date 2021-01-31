#ifndef BAYESMIX_UTILS_IO_UTILS_H_
#define BAYESMIX_UTILS_IO_UTILS_H_

#include <Eigen/Dense>

#define MAXBUFSIZE ((int)1e6)

namespace bayesmix {
//! Returns an Eigen Matrix after reading it from a file.
Eigen::MatrixXd read_eigen_matrix(const std::string &filename);
void write_matrix_to_file(const Eigen::MatrixXd &mat, std::string filename);
}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_IO_UTILS_H_
