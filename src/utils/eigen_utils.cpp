#include "eigen_utils.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

Eigen::MatrixXd bayesmix::vstack(const std::vector<Eigen::MatrixXd> &mats) {
  auto cnt_rows = [&](int curr, const Eigen::MatrixXd &mat) {
    return curr + mat.rows();
  };
  int nrows = std::accumulate(mats.begin(), mats.end(), 0, cnt_rows);
  int ncols = mats[0].cols();

  Eigen::MatrixXd out(nrows, ncols);

  int begin = 0;

  for (int i = 0; i < mats.size(); i++) {
    out.block(begin, 0, mats[i].rows(), ncols) = mats[i];
    begin += mats[i].rows();
  }

  return out;
}

void bayesmix::append_by_row(Eigen::MatrixXd *a, const Eigen::MatrixXd &b) {
  if (!a->rows()) {
    *a = b;
  } else if (!b.rows()) {
    return;
  } else {
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
    Eigen::MatrixXd out(a.rows() + b.rows(), a.cols());
    out << a, b;
    return out;
  }

}