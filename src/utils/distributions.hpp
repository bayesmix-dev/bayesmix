#ifndef BAYESMIX_UTILS_DISTRIBUTIONS_HPP_
#define BAYESMIX_UTILS_DISTRIBUTIONS_HPP_

#include <Eigen/Dense>
#include <random>

namespace bayesmix {
int categorical_rng(const Eigen::VectorXd &probas, std::mt19937_64 &rng,
                    int start = 0);
double multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                              const Eigen::VectorXd &mean,
                              const Eigen::MatrixXd &prec_chol,
                              double prec_logdet);
}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_DISTRIBUTIONS_HPP_
