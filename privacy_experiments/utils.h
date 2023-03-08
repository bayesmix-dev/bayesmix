#ifndef BAYESMIX_PRIVACY_EXPERIMENTS_UTILS_H_
#define BAYESMIX_PRIVACY_EXPERIMENTS_UTILS_H_

#include "src/algorithms/private_neal2.h"
#include "src/includes.h"

std::shared_ptr<PrivateNeal2> get_algo1d(std::string hier_params,
                                         std::string mix_params,
                                         std::string algo_params);

Eigen::MatrixXd get_cluster_mat(BaseCollector* coll, int ndata);

double cluster_entropy(const Eigen::VectorXd& clustering);

#endif  // BAYESMIX_PRIVACY_EXPERIMENTS_UTILS_H_
