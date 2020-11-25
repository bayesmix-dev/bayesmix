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

}  // namespace bayesmix