#include <Eigen/Dense>
#include <vector>

//! This file implements a few methods to manipulate groups of matrices, mainly
//! by joining different objects, as well as additional utilities for SPD
//! checking and grid creation.

namespace bayesmix {
//! Concatenates a vector of Eigen matrices along the rows
//! @param mats The matrices to be concatenated
//! @return     The resulting matrix
//! @throw      std::invalid argument if sizes mismatch
Eigen::MatrixXd vstack(const std::vector<Eigen::MatrixXd> &mats);

//! Concatenates two matrices by row, modifying the first matrix in-place
//! @throw std::invalid_argument if sizes mismatch
void append_by_row(Eigen::MatrixXd *const a, const Eigen::MatrixXd &b);

//! Concatenates two matrices by row
//! @param a,b The matrices to be concatenated
//! @return    The resulting matrix
//! @throw     std::invalid_argument if sizes mismatch
Eigen::MatrixXd append_by_row(const Eigen::MatrixXd &a,
                              const Eigen::MatrixXd &b);

//! Creates an Eigen matrix from a collection of rows
//! @tparam Container  An std-compatible container implementing `operator[]`
//! @param rows        The rows of the matrix
//! @return            The resulting matrix
template <template <typename...> class Container>
Eigen::MatrixXd stack_vectors(const Container<Eigen::VectorXd> &rows) {
  int nrows = rows.size();
  int ncols = rows[0].size();

  Eigen::MatrixXd out(nrows, ncols);
  for (int i = 0; i < nrows; i++) out.row(i) = rows[i].transpose();

  return out;
}

//! Checks whether the matrix is symmetric and semi-positive definite
void check_spd(const Eigen::MatrixXd &mat);

//! Creates a 2d grid over rectangle [x1, x2] x [y1, y2], with nx * ny points
//! @param x1, x2, y1, y2  Bounds for the rectangle
//! @param nx, ny          Number of points created along the x, y directions
//! @return                The resulting grid
Eigen::MatrixXd get_2d_grid(const double x1, const double x2, const int nx,
                            const double y1, const double y2, const int ny);
}  // namespace bayesmix
