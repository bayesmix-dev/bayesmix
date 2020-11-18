#include <Eigen/Dense>
#include <vector>

namespace bayesmix {
// Stacks a list of matrices by row, i.e. concatenating them one
// on top of the other
// TODO test
Eigen::MatrixXd vstack(const std::vector<Eigen::MatrixXd> &mats);

// TODO test
void append_by_row(Eigen::MatrixXd *a, const Eigen::MatrixXd &b);

}  // namespace bayesmix