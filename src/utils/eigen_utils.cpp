#include "eigen_utils.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

Eigen::MatrixXd bayesmix::vstack(const std::vector<Eigen::MatrixXd> &mats) {
  int ncols = mats[0].cols();
  for (int i = 0; i < mats.size(); i++) {
    if (mats[i].cols() != ncols) {
      std::stringstream msg;
      msg << "Expected all elements to have " << ncols << " columns, but entry"
          << " in position " << i << " has " << mats[i].cols() << " columns";
      throw std::invalid_argument(msg.str());
    }
  }

  auto cnt_rows = [&](int curr, const Eigen::MatrixXd &mat) {
    return curr + mat.rows();
  };
  int nrows = std::accumulate(mats.begin(), mats.end(), 0, cnt_rows);

  Eigen::MatrixXd out(nrows, ncols);
  int begin = 0;
  for (int i = 0; i < mats.size(); i++) {
    out.block(begin, 0, mats[i].rows(), ncols) = mats[i];
    begin += mats[i].rows();
  }

  return out;
}

void bayesmix::append_by_row(Eigen::MatrixXd *a, const Eigen::MatrixXd &b) {
  if (a->rows() == 0) {
    *a = b;
  } else if (b.rows() == 0) {
    return;
  } else {
    if (a->cols() != b.cols()) {
      std::stringstream msg;
      msg << "Expected a and b to have the same number of columns, but"
          << " a has shape " << a->rows() << "x" << a->cols() << ", while"
          << " b has shape " << b.rows() << "x" << b.cols();
      throw std::invalid_argument(msg.str());
    }
    int orig_rows = a->rows();
    a->conservativeResize(orig_rows + b.rows(), a->cols());
    a->block(orig_rows, 0, b.rows(), a->cols()) = b;
  }
}

Eigen::MatrixXd bayesmix::append_by_row(const Eigen::MatrixXd &a,
                                        const Eigen::MatrixXd &b) {
  if (a.rows() == 0)
    return b;
  else if (b.rows() == 0)
    return a;
  else {
    if (a.cols() != b.cols()) {
      std::stringstream msg;
      msg << "Expected a and b to have the same number of columns, but"
          << " a has shape " << a.rows() << "x" << a.cols() << ", while"
          << " b has shape " << b.rows() << "x" << b.cols();
      throw std::invalid_argument(msg.str());
    }
    Eigen::MatrixXd out(a.rows() + b.rows(), a.cols());
    out << a, b;
    return out;
  }
}

void bayesmix::check_spd(const Eigen::MatrixXd &mat) {
  assert(mat.rows() == mat.cols());
  assert(mat.isApprox(mat.transpose()) && "Error: matrix is not symmetric");
  stan::math::check_pos_definite("", "Matrix", mat);
}
