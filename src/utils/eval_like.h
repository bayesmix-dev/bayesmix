#ifndef BAYESMIX_UTILS_EVAL_LIKE_H_
#define BAYESMIX_UTILS_EVAL_LIKE_H_

#include "src/includes.h"
#include "src/utils/eigen_utils.h"

namespace bayesmix {

Eigen::MatrixXd eval_lpdf_parallel(std::shared_ptr<BaseAlgorithm> algo,
                                   BaseCollector *const collector,
                                   const Eigen::MatrixXd &grid,
                                   int chunk_size = 100);

}

#endif
