#ifndef BAYESMIX_UTILS_DISTRIBUTIONS_H_
#define BAYESMIX_UTILS_DISTRIBUTIONS_H_


#include <Eigen/Dense>
#include <random>
#include <vector>

#include "algorithm_state.pb.h"

namespace bayesmix {

/*
 * Return a pseudorandom categorical random varialbe on the set
 * {start, ...., start + k} where k is defined as the size of the
 * probability vector
 *
 * @param probas Probabilities for each category
 * @param rng random number generator
 * @param start (default = 0)
 * @return categorical random variate with values on {start, ...., start + k}
 */
int categorical_rng(const Eigen::VectorXd &probas, std::mt19937_64 &rng,
                    int start = 0);

/*
 * Evaluates the log probability density function of a multivariate Gaussian
 * distribution parametrized by mean and precision matrix
 *
 * @param datum where to evaluate the the lpdf
 * @param mean the mean of the Gaussian distribution
 * @prec_chol the (lower) cholesky factor of the precision matric
 * @prec_logdet logarithm of the determinant of the precision matrix
 * @return the evaluation of the lpdf
 */
double multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                              const Eigen::VectorXd &mean,
                              const Eigen::MatrixXd &prec_chol,
                              double prec_logdet);

/*
 * Evaluates the log probability density function of a multivariate Gaussian
 * distribution parametrized by mean and precision matrix
 *
 * @param data a grid of points (by row) where to evaluate the lpdf
 * @param mean the mean of the Gaussian distribution
 * @prec_chol the (lower) cholesky factor of the precision matric
 * @prec_logdet logarithm of the determinant of the precision matrix
 * @return the evaluation of the lpdf
 */
Eigen::VectorXd multi_normal_prec_lpdf_grid(const Eigen::MatrixXd &data,
                                            const Eigen::VectorXd &mean,
                                            const Eigen::MatrixXd &prec_chol,
                                            double prec_logdet);

double multi_student_t_invscale_lpdf(const Eigen::VectorXd &datum, double df,
                                     const Eigen::VectorXd &mean,
                                     const Eigen::MatrixXd &invscale_chol,
                                     double scale_logdet);

Eigen::VectorXd multi_student_t_invscale_lpdf_grid(
    const Eigen::MatrixXd &data, double df, const Eigen::VectorXd &mean,
    const Eigen::MatrixXd &invscale_chol, double scale_logdet);

/*
 * Computes the L2 distance between the univariate mixture of Gaussian
 * densities p1(x) = \sum_{h=1}^m1 w1[h] N(x | mean1[h], var1[h]) and
 * p2(x) = \sum_{h=1}^m2 w2[h] N(x | mean2[h], var2[h])
 *
 * The L2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 *
 * @param means1 the means of the first mixture density
 * @param vars1 the variances of the first mixture density
 * @param weights1 the weigths of the first mixture density
 * @param means2 the means of the second mixture density
 * @param vars2 the variances of the second mixture density
 * @param weights2 the weigths of the second mixture density
 * @return the L2 distance between p and q
 */
double gaussian_mixture_dist(Eigen::VectorXd means1, Eigen::VectorXd vars1,
                             Eigen::VectorXd weights1, Eigen::VectorXd means2,
                             Eigen::VectorXd vars2, Eigen::VectorXd weights2);

/*
 * Computes the L2 distance between the multivariate mixture of Gaussian
 * densities p1(x) = \sum_{h=1}^m1 w1[h] N(x | mean1[h], Prec[1]^{-1}) and
 * p2(x) = \sum_{h=1}^m2 w2[h] N(x | mean2[h], Prec2[h]^{-1})
 *
 * The L2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 *
 * @param means1 the means of the first mixture density
 * @param precs1 the precisions of the first mixture density
 * @param weights1 the weigths of the first mixture density
 * @param means2 the means of the second mixture density
 * @param precs2 the precision of the second mixture density
 * @param weights2 the weigths of the second mixture density
 * @return the L2 distance between p and q
 */
double gaussian_mixture_dist(std::vector<Eigen::VectorXd> means1,
                             std::vector<Eigen::MatrixXd> precs1,
                             Eigen::VectorXd weights1,
                             std::vector<Eigen::VectorXd> means2,
                             std::vector<Eigen::MatrixXd> precs2,
                             Eigen::VectorXd weights2);
/*
 * Computes the L2 distance between the mixture of Gaussian
 * densities p1(x) and p2. These could be either univariate or multivariate
 * The L2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 *
 * @param clus1 the cluster-specific parameters of the first mixture density
 * @param weights1 the weigths of the first mixture density
 * @param clus2 the cluster-specific parameters of the second mixture density
 * @param weights2 the weigths of the second mixture density
 * @return the L2 distance between p and q
 */
double gaussian_mixture_dist(
    std::vector<bayesmix::AlgorithmState::ClusterState> clus1,
    Eigen::VectorXd weights1,
    std::vector<bayesmix::AlgorithmState::ClusterState> clus2,
    Eigen::VectorXd weights2);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_DISTRIBUTIONS_H_
