#ifndef BAYESMIX_SRC_UTILS_COVARIATES_GETTER_H
#define BAYESMIX_SRC_UTILS_COVARIATES_GETTER_H

#include <Eigen/Dense>

class covariates_getter {
 protected:
  const Eigen::MatrixXd* covariates;

 public:
  covariates_getter(const Eigen::MatrixXd& covariates_)
      : covariates(&covariates_){};

  Eigen::RowVectorXd operator()(const size_t& i) const {
    if (covariates->cols() == 0) {
      return Eigen::RowVectorXd(0);
    } else if (covariates->rows() == 1) {
      return covariates->row(0);
    } else {
      return covariates->row(i);
    }
  };
};

#endif  // BAYESMIX_SRC_UTILS_COVARIATES_GETTER_H
