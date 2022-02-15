#include "eigen_utils.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stan/math/prim/err.hpp>

Eigen::MatrixXd bayesmix::vstack(const std::vector<Eigen::MatrixXd> &mats) {
  // Check that row dimensions are consistent
  int ncols = mats[0].cols();
  for (int i = 0; i < mats.size(); i++) {
    if (mats[i].cols() != ncols) {
      std::stringstream msg;
      msg << "Expected all elements to have " << ncols << " columns, but entry"
          << " in position " << i << " has " << mats[i].cols() << " columns";
      throw std::invalid_argument(msg.str());
    }
  }

  // Counts total number of rows
  auto cnt_rows = [&](int curr, const Eigen::MatrixXd &mat) {
    return curr + mat.rows();
  };
  int nrows = std::accumulate(mats.begin(), mats.end(), 0, cnt_rows);

  // Write new matrix block by block
  Eigen::MatrixXd out(nrows, ncols);
  int begin = 0;
  for (int i = 0; i < mats.size(); i++) {
    out.block(begin, 0, mats[i].rows(), ncols) = mats[i];
    begin += mats[i].rows();
  }

  return out;
}

void bayesmix::append_by_row(Eigen::MatrixXd *const a,
                             const Eigen::MatrixXd &b) {
  if (a->rows() == 0) {
    *a = b;
  } else if (b.rows() == 0) {
    return;
  } else {
    // Check that column dimensions are consistent
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
    // Check that column dimensions are consistent
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
  if (mat.rows() != mat.cols()) {
    throw std::invalid_argument("Matrix is not square");
  }
  if (mat.isApprox(mat.transpose()) == false) {
    throw std::invalid_argument("Matrix is not symmetric");
  }
  stan::math::check_pos_definite("", "Matrix", mat);
}

Eigen::MatrixXd bayesmix::get_2d_grid(const double x1, const double x2,
                                      const int nx, const double y1,
                                      const double y2, const int ny) {
  Eigen::VectorXd xgrid = Eigen::ArrayXd::LinSpaced(nx, x1, x2);
  Eigen::VectorXd ygrid = Eigen::ArrayXd::LinSpaced(ny, y1, y2);
  Eigen::MatrixXd out(nx * ny, 2);
  for (int i = 0; i < xgrid.size(); i++) {
    for (int j = 0; j < ygrid.size(); j++) {
      Eigen::VectorXd curr(2);
      curr << xgrid(i), ygrid(j);
      out.row(i * xgrid.size() + j) = curr;
    }
  }
  return out;
}
