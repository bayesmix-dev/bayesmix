#include <Eigen/Dense>
#include <vector>

namespace bayesmix {

/*
 * Concatenates a vector of Eigen Matrices along the rows
 * @param mats std::vector<Eigen::MatrixXd> the matrices to concatenate
 * @return Eigen::MatrixXd
 * @ throw std::invalid argument if sizes mismatch
 */
Eigen::MatrixXd vstack(const std::vector<Eigen::MatrixXd> &mats);

/*
 * Concatenates two matrices by row, modifying the first matrix in-place
 * @param a pointer to Eigen::MatrixXd the first matrix (will be overwritten)
 * @param b Eigen::MatrixXd the second matrix
 * @throws std::invalid_argument if sizes mismatch
 */
void append_by_row(Eigen::MatrixXd *a, const Eigen::MatrixXd &b);

/*
 * Concatenates two matrices by row
 * @param a Eigen::MatrixXd the first matrix
 * @param b Eigen::MatrixXd the second matrix
 * @return Eigen::MatrixXd the concatenated matrix
 * @throws std::invalid_argument if sizes mismatch
 */
Eigen::MatrixXd append_by_row(const Eigen::MatrixXd &a,
                              const Eigen::MatrixXd &b);

/*
 * Creates an Eigen::Matrix from a collection of rows
 * @tparam Container an std-compatible container implementing the square
 * brackets access operator
 * @param rows the rows of the matrix
 */

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

/*
 * creates a 2d grid over the rectangle [x1, x2] x [y1, y2]
 * by expanding the grids covering [x1, x2] and [y1, y2] with nx and ny points
 */
Eigen::MatrixXd get_2d_grid(double x1, double x2, int nx, double y1, double y2,
                            int ny);

}  // namespace bayesmix
