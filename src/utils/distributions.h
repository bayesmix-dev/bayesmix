#ifndef BAYESMIX_UTILS_DISTRIBUTIONS_H_
#define BAYESMIX_UTILS_DISTRIBUTIONS_H_


#include <Eigen/Dense>
#include <random>
#include <vector>

#include "algorithm_state.pb.h"

//! This file includes several useful functions related to probability
//! distributions, including categorical variables, popular multivariate
//! distributions, and distribution distances. Some of these functions make use
//! of OpenMP parallelism to achieve better efficiency.

namespace bayesmix {

/*
 * Returns a pseudorandom categorical random variable on the set
 * {start, ..., start + k} where k is the size of the given probability vector
 *
 * @param probas Probabilities for each category
 * @param rng    Random number generator
 * @param start  (default = 0)
 * @return       categorical r.v. with values on {start, ..., start + k}
 */
int categorical_rng(const Eigen::VectorXd &probas, std::mt19937_64 &rng,
                    int start = 0);

/*
 * Evaluates the log probability density function of a multivariate Gaussian
 * distribution parametrized by mean and precision matrix on a single point
 *
 * @param datum  Point in which to evaluate the the lpdf
 * @param mean   The mean of the Gaussian distribution
 * @prec_chol    The (lower) Cholesky factor of the precision matrix
 * @prec_logdet  The logarithm of the determinant of the precision matrix
 * @return       The evaluation of the lpdf
 */
double multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                              const Eigen::VectorXd &mean,
                              const Eigen::MatrixXd &prec_chol,
                              double prec_logdet);

/*
 * Evaluates the log probability density function of a multivariate Gaussian
 * distribution parametrized by mean and precision matrix on multiple points
 *
 * @param data   Grid of points (by row) on which to evaluate the lpdf
 * @param mean   The mean of the Gaussian distribution
 * @prec_chol    The (lower) Cholesky factor of the precision matrix
 * @prec_logdet  The logarithm of the determinant of the precision matrix
 * @return       The evaluation of the lpdf
 */
Eigen::VectorXd multi_normal_prec_lpdf_grid(const Eigen::MatrixXd &data,
                                            const Eigen::VectorXd &mean,
                                            const Eigen::MatrixXd &prec_chol,
                                            double prec_logdet);

/*
 * Evaluates the log probability density function of a multivariate Student's t
 * distribution on a single point
 *
 * @param datum    Point in which to evaluate the the lpdf
 * @param df       The degrees of freedom of the Student's t distribution
 * @param mean     The mean of the Student's t distribution
 * @invscale_chol  The (lower) Cholesky factor of the inverse scale matrix
 * @prec_logdet    The logarithm of the determinant of the inverse scale matrix
 * @return         The evaluation of the lpdf
 */
double multi_student_t_invscale_lpdf(const Eigen::VectorXd &datum, double df,
                                     const Eigen::VectorXd &mean,
                                     const Eigen::MatrixXd &invscale_chol,
                                     double scale_logdet);

/*
 * Evaluates the log probability density function of a multivariate Student's t
 * distribution on multiple points
 *
 * @param data     Grid of points (by row) on which to evaluate the lpdf
 * @param df       The degrees of freedom of the Student's t distribution
 * @param mean     The mean of the Student's t distribution
 * @invscale_chol  The (lower) Cholesky factor of the inverse scale matrix
 * @prec_logdet    The logarithm of the determinant of the inverse scale matrix
 * @return         The evaluation of the lpdf
 */
Eigen::VectorXd multi_student_t_invscale_lpdf_grid(
    const Eigen::MatrixXd &data, double df, const Eigen::VectorXd &mean,
    const Eigen::MatrixXd &invscale_chol, double scale_logdet);

/*
 * Computes the L^2 distance between the univariate mixture of Gaussian
 * densities p1(x) = \sum_{h=1}^m1 w1[h] N(x | mean1[h], var1[h]) and
 * p2(x) = \sum_{h=1}^m2 w2[h] N(x | mean2[h], var2[h])
 *
 * The L^2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 */
double gaussian_mixture_dist(Eigen::VectorXd means1, Eigen::VectorXd vars1,
                             Eigen::VectorXd weights1, Eigen::VectorXd means2,
                             Eigen::VectorXd vars2, Eigen::VectorXd weights2);

/*
 * Computes the L^2 distance between the multivariate mixture of Gaussian
 * densities p1(x) = \sum_{h=1}^m1 w1[h] N(x | mean1[h], Prec[1]^{-1}) and
 * p2(x) = \sum_{h=1}^m2 w2[h] N(x | mean2[h], Prec2[h]^{-1})
 *
 * The L^2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 */
double gaussian_mixture_dist(std::vector<Eigen::VectorXd> means1,
                             std::vector<Eigen::MatrixXd> precs1,
                             Eigen::VectorXd weights1,
                             std::vector<Eigen::VectorXd> means2,
                             std::vector<Eigen::MatrixXd> precs2,
                             Eigen::VectorXd weights2);

/*
 * Computes the L^2 distance between the mixture of Gaussian
 * densities p(x) and q(x). These could be either univariate or multivariate.
 * The L2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 *
 * @param clus1, clus2        Cluster-specific parameters of the mix. densities
 * @param weights1, weights2  Weigths of the mixture densities
 * @return                    The L^2 distance between p and q
 */
double gaussian_mixture_dist(
    std::vector<bayesmix::AlgorithmState::ClusterState> clus1,
    Eigen::VectorXd weights1,
    std::vector<bayesmix::AlgorithmState::ClusterState> clus2,
    Eigen::VectorXd weights2);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_DISTRIBUTIONS_H_
