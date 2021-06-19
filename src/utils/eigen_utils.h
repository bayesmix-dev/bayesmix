#include <Eigen/Dense>
#include <vector>

namespace bayesmix {
//! Concatenates a vector of Eigen matrices along the rows
//! @param mats The matrices to be concatenated
//! @return     The resulting matrix
//! @throw      std::invalid argument if sizes mismatch
Eigen::MatrixXd vstack(const std::vector<Eigen::MatrixXd> &mats);

//! Concatenates two matrices by row, modifying the first matrix in-place
//! @throw std::invalid_argument if sizes mismatch
void append_by_row(Eigen::MatrixXd *a, const Eigen::MatrixXd &b);

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
Eigen::MatrixXd get_2d_grid(double x1, double x2, int nx, double y1, double y2,
                            int ny);
}  // namespace bayesmix
