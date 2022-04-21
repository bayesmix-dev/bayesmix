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
int categorical_rng(const Eigen::VectorXd &probas, std::mt19937 &rng,
                    const int start = 0);

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
                              const double prec_logdet);

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
                                            const double prec_logdet);

/*
 * Returns a pseudorandom multivariate normal random variable with diagonal
 * covariance matrix
 *
 * @param mean   The mean of the Gaussian r.v.
 * @param cov_diag   The diagonal covariance matrix
 * @rng          Random number generator
 * @return       multivariate normal r.v.
 */
Eigen::VectorXd multi_normal_diag_rng(
    const Eigen::VectorXd &mean,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &cov_diag,
    std::mt19937 &rng);

/*
 * Returns a pseudorandom multivariate normal random variable parametrized
 * through mean and Cholesky decomposition of precision matrix
 *
 * @param mean   The mean of the Gaussian r.v.
 * @prec_chol    The (lower) Cholesky factor of the precision matrix
 * @param rng    Random number generator
 * @return       multivariate normal r.v.
 */
Eigen::VectorXd multi_normal_prec_chol_rng(
    const Eigen::VectorXd &mean, const Eigen::LLT<Eigen::MatrixXd> &prec_chol,
    std::mt19937 &rng);

/*
 * Evaluates the log probability density function of a multivariate Gaussian
 * distribution with the following covariance structure:
 * Sigma + Lambda * Lambda^T
 * where Sigma is a diagonal matrix and Lambda a (p x d) one. Usually, p >> d.
 * y^T*(Sigma + Lambda * Lambda^T)^{-1}*y = y^T*Sigma^{-1}*y -
 * ||wood_factor*y||^2
 *
 *
 * @param datum                 Point on which to evaluate the lpdf
 * @param mean                  The mean of the Gaussian distribution
 * @param sigma_diag_inverse    The inverse of the diagonal of Sigma matrix
 * @param wood_factor           Computed as L^{-1} * Lambda^T * Sigma^{-1},
 * where L is the (lower) Cholesky factor of I + Lambda^T * Sigma^{-1} * Lambda
 * @param cov_logdet            The logarithm of the determinant of the
 * covariance matrix
 * @return                      The evaluation of the lpdf
 */
double multi_normal_lpdf_woodbury_chol(
    const Eigen::RowVectorXd &datum, const Eigen::VectorXd &mean,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &sigma_diag_inverse,
    const Eigen::MatrixXd &wood_factor, const double &cov_logdet);

/*
 * Evaluates the log probability density function of a multivariate Gaussian
 * distribution with the following covariance structure:
 * Sigma + Lambda * Lambda^T
 * where Sigma is a diagonal matrix and Lambda a (p x d) one. Usually, p >> d.
 * The Woodbury matrix identity
 * (https://en.wikipedia.org/wiki/Woodbury_matrix_identity) is used to turn
 * computation from being O(p^3) to being O(d^3 p) which gives a substantial
 * speedup when p >> d
 *
 * @param datum  Point on which to evaluate the lpdf
 * @param mean   The mean of the Gaussian distribution
 * @param sigma_diag   The diagonal of Sigma matrix
 * @param lambda       Rectangular matrix in Woodbury Identity
 * @return       The evaluation of the lpdf
 */
double multi_normal_lpdf_woodbury(const Eigen::VectorXd &datum,
                                  const Eigen::VectorXd &mean,
                                  const Eigen::VectorXd &sigma_diag,
                                  const Eigen::MatrixXd &lambda);

/*
 * Returns the log-determinant of the matrix Lambda Lambda^T + Sigma
 * and the 'wood_factor', i.e.
 *           L^{-1} * Lambda^T * Sigma^{-1},
 * where L is the (lower) Cholesky factor of
 *          I + Lambda^T * Sigma^{-1} * Lambda
 *
 * @param sigma_dag_inverse The inverse of the diagonal matrix Sigma
 * @param lambda            The matrix Lambda
 */
std::pair<Eigen::MatrixXd, double> compute_wood_chol_and_logdet(
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &sigma_diag_inverse,
    const Eigen::MatrixXd &lambda);

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
double multi_student_t_invscale_lpdf(const Eigen::VectorXd &datum,
                                     const double df,
                                     const Eigen::VectorXd &mean,
                                     const Eigen::MatrixXd &invscale_chol,
                                     const double scale_logdet);

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
    const Eigen::MatrixXd &data, const double df, const Eigen::VectorXd &mean,
    const Eigen::MatrixXd &invscale_chol, const double scale_logdet);

/*
 * Computes the L^2 distance between the univariate mixture of Gaussian
 * densities p1(x) = \sum_{h=1}^m1 w1[h] N(x | mean1[h], var1[h]) and
 * p2(x) = \sum_{h=1}^m2 w2[h] N(x | mean2[h], var2[h])
 *
 * The L^2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 */
double gaussian_mixture_dist(const Eigen::VectorXd &means1,
                             const Eigen::VectorXd &vars1,
                             const Eigen::VectorXd &weights1,
                             const Eigen::VectorXd &means2,
                             const Eigen::VectorXd &vars2,
                             const Eigen::VectorXd &weights2);

/*
 * Computes the L^2 distance between the multivariate mixture of Gaussian
 * densities p1(x) = \sum_{h=1}^m1 w1[h] N(x | mean1[h], Prec[1]^{-1}) and
 * p2(x) = \sum_{h=1}^m2 w2[h] N(x | mean2[h], Prec2[h]^{-1})
 *
 * The L^2 distance amounts to
 * d(p, q) = (\int (p(x) - q(x)^2 dx))^{1/2}
 */
double gaussian_mixture_dist(const std::vector<Eigen::VectorXd> &means1,
                             const std::vector<Eigen::MatrixXd> &precs1,
                             const Eigen::VectorXd &weights1,
                             const std::vector<Eigen::VectorXd> &means2,
                             const std::vector<Eigen::MatrixXd> &precs2,
                             const Eigen::VectorXd &weights2);

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
    const std::vector<bayesmix::AlgorithmState::ClusterState> &clus1,
    const Eigen::VectorXd &weights1,
    const std::vector<bayesmix::AlgorithmState::ClusterState> &clus2,
    const Eigen::VectorXd &weights2);

}  // namespace bayesmix

#endif  // BAYESMIX_UTILS_DISTRIBUTIONS_H_
