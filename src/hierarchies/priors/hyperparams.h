#ifndef BAYESMIX_HIERARCHIES_PRIORMODEL_HYPERPARAMS_H_
#define BAYESMIX_HIERARCHIES_PRIORMODEL_HYPERPARAMS_H_

#include <Eigen/Dense>

namespace Hyperparams {

struct NIG {
  double mean, var_scaling, shape, scale;
};

struct NxIG {
  double mean, var, shape, scale;
};

struct NW {
  Eigen::VectorXd mean;
  double var_scaling, deg_free;
  Eigen::MatrixXd scale, scale_inv, scale_chol;
};

struct MNIG {
  Eigen::VectorXd mean;
  Eigen::MatrixXd var_scaling, var_scaling_inv;
  double shape, scale;
};

}  // namespace Hyperparams

#endif  // BAYESMIX_HIERARCHIES_PRIORMODEL_HYPERPARAMS_H_
