#ifndef BAYESMIX_UTILS_DISTRIBUTIONS_HPP_
#define BAYESMIX_UTILS_DISTRIBUTIONS_HPP_

#include <omp.h>
#include <proto/cpp/marginal_state.pb.h>

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace bayesmix {

int categorical_rng(const Eigen::VectorXd &probas, std::mt19937_64 &rng,
                    int start = 0);
double multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                              const Eigen::VectorXd &mean,
                              const Eigen::MatrixXd &prec_chol,
                              double prec_logdet);

double gaussian_mixture_dist(Eigen::VectorXd means1, Eigen::VectorXd sds1,
                             Eigen::VectorXd weights1, Eigen::VectorXd means2,
                             Eigen::VectorXd sds2, Eigen::VectorXd weights2);

double gaussian_mixture_dist(std::vector<Eigen::VectorXd> means1,
                             std::vector<Eigen::MatrixXd> precs1,
                             Eigen::VectorXd weights1,
                             std::vector<Eigen::VectorXd> means2,
                             std::vector<Eigen::MatrixXd> precs2,
                             Eigen::VectorXd weights2);

double gaussian_mixture_dist(
    std::vector<bayesmix::MarginalState::ClusterState> clus1,
    Eigen::VectorXd weights1,
    std::vector<bayesmix::MarginalState::ClusterState> clus2,
    Eigen::VectorXd weights2);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_DISTRIBUTIONS_HPP_
